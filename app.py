#!/usr/bin/env python
# coding=utf-8
"""
VibeVoice ASR Portable - Русскоязычная версия
Распознавание речи в текст с поддержкой 4-bit квантизации

Авторы:
@Nerual Dreming - портативная версия, русификация
Нейро-Софт (neuro-cartel.com) - адаптация и поддержка
Microsoft - оригинальная модель VibeVoice ASR
"""

import os
import sys
import tempfile

# GRADIO_TEMP_DIR устанавливается в run.bat

# Добавляем директорию скрипта в sys.path для импорта локального модуля vibevoice
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)
import torch
import numpy as np
import soundfile as sf
from pathlib import Path
import time
import json
import gradio as gr
from typing import List, Dict, Tuple, Optional, Generator
import tempfile
import base64
import io
import traceback
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc
import re
import shutil
import uuid

from transformers import TextIteratorStreamer, StoppingCriteria, StoppingCriteriaList, BitsAndBytesConfig

try:
    from pydub import AudioSegment
    HAS_PYDUB = True
except ImportError:
    HAS_PYDUB = False
    print("Предупреждение: pydub не установлен, используется WAV формат")

from vibevoice.modular.modeling_vibevoice_asr import VibeVoiceASRForConditionalGeneration
from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor
from vibevoice.processor.audio_utils import load_audio_use_ffmpeg, COMMON_AUDIO_EXTS

APP_VERSION = "1.0.0"
APP_NAME = "VibeVoice ASR Portable"

SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "output"
TEMP_DIR = SCRIPT_DIR / "temp"

OUTPUT_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)

AVAILABLE_MODELS = [
    ("VibeVoice ASR (полная модель)", "microsoft/VibeVoice-ASR"),
    ("VibeVoice ASR 4-bit (экономия памяти)", "scerz/VibeVoice-ASR-4bit"),
]


class VibeVoiceASRInference:
    """Класс для инференса VibeVoice ASR модели."""
    
    def __init__(
        self, 
        model_path: str, 
        device: str = "cuda", 
        dtype: torch.dtype = torch.bfloat16, 
        attn_implementation: str = "flash_attention_2",
        use_4bit: bool = False
    ):
        """
        Инициализация ASR пайплайна.
        
        Args:
            model_path: Путь к модели или имя на HuggingFace
            device: Устройство для инференса
            dtype: Тип данных для весов модели
            attn_implementation: Реализация attention ('flash_attention_2', 'sdpa', 'eager')
            use_4bit: Использовать 4-bit квантизацию
        """
        print(f"Загрузка модели VibeVoice ASR из {model_path}")
        
        self.processor = VibeVoiceASRProcessor.from_pretrained(model_path)
        
        print(f"Используется attention: {attn_implementation}")
        
        model_kwargs = {
            "dtype": dtype,
            "attn_implementation": attn_implementation,
            "trust_remote_code": True
        }
        
        is_prequantized = "4bit" in model_path.lower() or "4-bit" in model_path.lower()
        use_device_map = is_prequantized or use_4bit
        
        if is_prequantized:
            print("Модель уже квантизирована (4-bit), пропускаем дополнительную квантизацию")
            model_kwargs["device_map"] = "auto"
        elif use_4bit:
            print("Включена 4-bit квантизация (bitsandbytes)")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            model_kwargs["quantization_config"] = quantization_config
            model_kwargs["device_map"] = "auto"
        else:
            print("Загрузка полной модели в bfloat16")
            model_kwargs["device_map"] = device if device == "auto" else None
        
        self.model = VibeVoiceASRForConditionalGeneration.from_pretrained(
            model_path,
            **model_kwargs
        )
        
        if not use_device_map and device != "auto":
            self.model = self.model.to(device)
        
        self.device = device if device != "auto" else next(self.model.parameters()).device
        self.model.eval()
        
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Модель загружена на {self.device}")
        print(f"Параметров: {total_params:,} ({total_params/1e9:.2f}B)")
        
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"GPU память: {allocated:.2f} GB выделено, {reserved:.2f} GB зарезервировано")
    
    def transcribe(
        self, 
        audio_path: str = None,
        audio_array: np.ndarray = None,
        sample_rate: int = None,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
        do_sample: bool = False,
        num_beams: int = 1,
        repetition_penalty: float = 1.0,
        context_info: str = None,
        streamer: Optional[TextIteratorStreamer] = None,
    ) -> dict:
        """
        Транскрибация аудио в текст.
        """
        inputs = self.processor(
            audio=audio_path,
            sampling_rate=sample_rate,
            return_tensors="pt",
            add_generation_prompt=True,
            context_info=context_info
        )
        
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                  for k, v in inputs.items()}
        
        generation_config = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature if temperature > 0 else None,
            "top_p": top_p if do_sample else None,
            "do_sample": do_sample,
            "num_beams": num_beams,
            "repetition_penalty": repetition_penalty,
            "pad_token_id": self.processor.pad_id,
            "eos_token_id": self.processor.tokenizer.eos_token_id,
        }
        
        if streamer is not None:
            generation_config["streamer"] = streamer
        
        generation_config["stopping_criteria"] = StoppingCriteriaList([StopOnFlag()])
        generation_config = {k: v for k, v in generation_config.items() if v is not None}
        
        start_time = time.time()
        
        input_ids = inputs['input_ids'][0]
        total_input_tokens = input_ids.shape[0]
        
        pad_id = self.processor.pad_id
        padding_mask = (input_ids == pad_id)
        num_padding_tokens = padding_mask.sum().item()
        
        speech_start_id = self.processor.speech_start_id
        speech_end_id = self.processor.speech_end_id
        
        input_ids_list = input_ids.tolist()
        num_speech_tokens = 0
        in_speech = False
        for token_id in input_ids_list:
            if token_id == speech_start_id:
                in_speech = True
                num_speech_tokens += 1
            elif token_id == speech_end_id:
                in_speech = False
                num_speech_tokens += 1
            elif in_speech:
                num_speech_tokens += 1
        
        num_text_tokens = total_input_tokens - num_speech_tokens - num_padding_tokens
        
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                **generation_config
            )
        
        generation_time = time.time() - start_time
        
        generated_ids = output_ids[0, inputs['input_ids'].shape[1]:]
        generated_text = self.processor.decode(generated_ids, skip_special_tokens=True)
        
        try:
            transcription_segments = self.processor.post_process_transcription(generated_text)
        except Exception as e:
            print(f"Предупреждение: Не удалось распарсить вывод: {e}")
            transcription_segments = []
        
        return {
            "raw_text": generated_text,
            "segments": transcription_segments,
            "generation_time": generation_time,
            "input_tokens": {
                "total": total_input_tokens,
                "speech": num_speech_tokens,
                "text": num_text_tokens,
                "padding": num_padding_tokens,
            },
        }


asr_model = None
stop_generation_event = threading.Event()  # Потокобезопасный флаг остановки
current_model_path = None
model_loading_lock = threading.Lock()


class StopOnFlag(StoppingCriteria):
    """Критерий остановки по потокобезопасному событию."""
    def __call__(self, input_ids, scores, **kwargs):
        return stop_generation_event.is_set()


def parse_time_to_seconds(val: Optional[str]) -> Optional[float]:
    """Парсинг времени в секунды."""
    if val is None:
        return None
    val = val.strip()
    if not val:
        return None
    try:
        return float(val)
    except ValueError:
        pass
    if ":" in val:
        parts = val.split(":")
        if not all(p.strip().replace(".", "", 1).isdigit() for p in parts):
            return None
        parts = [float(p) for p in parts]
        if len(parts) == 3:
            h, m, s = parts
        elif len(parts) == 2:
            h = 0
            m, s = parts
        else:
            return None
        return h * 3600 + m * 60 + s
    return None


def slice_audio_to_temp(
    audio_data: np.ndarray,
    sample_rate: int,
    start_sec: Optional[float],
    end_sec: Optional[float]
) -> Tuple[Optional[str], Optional[str]]:
    """Нарезка аудио и сохранение во временный файл."""
    n_samples = len(audio_data)
    full_duration = n_samples / float(sample_rate)
    start = 0.0 if start_sec is None else max(0.0, start_sec)
    end = full_duration if end_sec is None else min(full_duration, end_sec)
    if end <= start:
        return None, f"Неверный диапазон времени: начало={start:.2f}с, конец={end:.2f}с"
    start_idx = int(start * sample_rate)
    end_idx = int(end * sample_rate)
    segment = audio_data[start_idx:end_idx]
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav", dir=str(TEMP_DIR))
    temp_file.close()
    segment_int16 = (segment * 32768.0).astype(np.int16)
    sf.write(temp_file.name, segment_int16, sample_rate, subtype='PCM_16')
    return temp_file.name, None


def get_device():
    """Определение устройства."""
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_attn_implementation():
    """Определение лучшей реализации attention."""
    if not torch.cuda.is_available():
        return "eager"
    try:
        import flash_attn
        return "flash_attention_2"
    except ImportError:
        return "sdpa"


def initialize_model(model_path: str, use_4bit: bool = False):
    """Инициализация модели ASR с защитой от двойных кликов."""
    global asr_model, current_model_path
    
    if not model_loading_lock.acquire(blocking=False):
        return "Загрузка уже выполняется, подождите..."
    
    try:
        # Очистка предыдущей модели из памяти
        if asr_model is not None:
            print("Выгрузка предыдущей модели из памяти...")
            del asr_model
            asr_model = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            print("Память очищена")
        
        device = get_device()
        dtype = torch.bfloat16 if device != "cpu" else torch.float32
        attn_impl = get_attn_implementation()
        
        asr_model = VibeVoiceASRInference(
            model_path=model_path,
            device=device,
            dtype=dtype,
            attn_implementation=attn_impl,
            use_4bit=use_4bit
        )
        current_model_path = model_path
        return f"Модель загружена: {model_path}"
    except Exception as e:
        traceback.print_exc()
        return f"Ошибка загрузки модели: {str(e)}"
    finally:
        model_loading_lock.release()


def clip_and_encode_audio(
    audio_data: np.ndarray,
    sr: int,
    start_time: float,
    end_time: float,
    segment_idx: int,
    use_mp3: bool = True,
    target_sr: int = 16000,
    mp3_bitrate: str = "32k"
) -> Tuple[int, Optional[str], Optional[str]]:
    """Нарезка и кодирование аудио сегмента в base64."""
    try:
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        
        start_sample = max(0, start_sample)
        end_sample = min(len(audio_data), end_sample)
        
        if start_sample >= end_sample:
            return segment_idx, None, f"Неверный диапазон: [{start_time:.2f}с - {end_time:.2f}с]"
        
        segment_data = audio_data[start_sample:end_sample]
        
        if sr != target_sr and target_sr < sr:
            duration = len(segment_data) / sr
            new_length = int(duration * target_sr)
            indices = np.linspace(0, len(segment_data) - 1, new_length)
            segment_data = np.interp(indices, np.arange(len(segment_data)), segment_data)
            sr = target_sr
        
        segment_data_int16 = (segment_data * 32768.0).astype(np.int16)
        
        if use_mp3 and HAS_PYDUB:
            try:
                wav_buffer = io.BytesIO()
                sf.write(wav_buffer, segment_data_int16, sr, format='WAV', subtype='PCM_16')
                wav_buffer.seek(0)
                
                audio_segment = AudioSegment.from_wav(wav_buffer)
                if audio_segment.channels > 1:
                    audio_segment = audio_segment.set_channels(1)
                mp3_buffer = io.BytesIO()
                audio_segment.export(mp3_buffer, format='mp3', bitrate=mp3_bitrate)
                mp3_buffer.seek(0)
                
                audio_bytes = mp3_buffer.read()
                audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
                audio_src = f"data:audio/mp3;base64,{audio_base64}"
                
                return segment_idx, audio_src, None
            except Exception as e:
                print(f"Ошибка конвертации в MP3 для сегмента {segment_idx}: {e}")
        
        wav_buffer = io.BytesIO()
        sf.write(wav_buffer, segment_data_int16, sr, format='WAV', subtype='PCM_16')
        wav_buffer.seek(0)
        
        audio_bytes = wav_buffer.read()
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        audio_src = f"data:audio/wav;base64,{audio_base64}"
        
        return segment_idx, audio_src, None
        
    except Exception as e:
        return segment_idx, None, f"Ошибка сегмента {segment_idx}: {str(e)}"


def extract_audio_segments(audio_path: str, segments: List[Dict]) -> List[Tuple[str, str, Optional[str]]]:
    """Извлечение аудио сегментов с параллельной обработкой."""
    try:
        print(f"Загрузка аудио: {audio_path}")
        max_retries = 5
        retry_delay = 0.5
        audio_data = None
        sr = None
        for attempt in range(max_retries):
            try:
                audio_data, sr = load_audio_use_ffmpeg(audio_path, resample=False)
                break
            except PermissionError as perm_err:
                if attempt < max_retries - 1:
                    print(f"PermissionError при загрузке аудио для сегментов (попытка {attempt + 1}/{max_retries}): {perm_err}")
                    time.sleep(retry_delay)
                else:
                    print(f"Не удалось загрузить аудио после {max_retries} попыток: {perm_err}")
                    return []
        if audio_data is None:
            return []
        print(f"Аудио загружено: {len(audio_data)} сэмплов, {sr} Гц")
        
        tasks = []
        use_mp3 = HAS_PYDUB
        
        for i, seg in enumerate(segments):
            start_time = seg.get('start_time')
            end_time = seg.get('end_time')
            
            if (not isinstance(start_time, (int, float)) or 
                not isinstance(end_time, (int, float)) or 
                start_time >= end_time):
                tasks.append((i, None, None, None, None, None))
                continue
            
            tasks.append((audio_data, sr, start_time, end_time, i, use_mp3))
        
        results = []
        max_workers = os.cpu_count() or 4
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for task in tasks:
                if task[0] is None:
                    continue
                future = executor.submit(clip_and_encode_audio, *task)
                futures[future] = task[4]
            
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    idx = futures[future]
                    results.append((idx, None, f"Ошибка обработки: {str(e)}"))
        
        results.sort(key=lambda x: x[0])
        
        audio_segments = []
        for i, (idx, audio_src, error_msg) in enumerate(results):
            seg = segments[idx] if idx < len(segments) else {}
            start_time = seg.get('start_time', 'N/A')
            end_time = seg.get('end_time', 'N/A')
            speaker_id = seg.get('speaker_id', 'N/A')
            
            segment_label = f"Сегмент {idx+1}: [{start_time:.2f}с - {end_time:.2f}с] Спикер {speaker_id}"
            audio_segments.append((segment_label, audio_src, error_msg))
        
        return audio_segments
        
    except Exception as e:
        print(f"Ошибка загрузки аудио: {e}")
        return []


def transcribe_audio(
    audio_file_input,
    video_file_input,
    mic_input,
    audio_path_input: str,
    start_time_input: str,
    end_time_input: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
    repetition_penalty: float = 1.0,
    context_info: str = "",
    model_path: str = None,
    use_4bit: bool = False
) -> Generator[Tuple[str, str], None, None]:
    """
    Транскрибация аудио с потоковым выводом.
    """
    global asr_model
    
    if asr_model is None:
        if model_path:
            yield "Загрузка модели...", ""
            result = initialize_model(model_path, use_4bit)
            if "Ошибка" in result:
                yield result, ""
                return
        else:
            yield "Ошибка: Модель не загружена! Выберите модель и нажмите 'Загрузить модель'.", ""
            return
    
    audio_input = audio_file_input or video_file_input or mic_input
    
    if audio_input is None:
        yield "Ошибка: Загрузите аудио/видео файл или запишите с микрофона!", ""
        return
    
    try:
        start_sec = parse_time_to_seconds(start_time_input)
        end_sec = parse_time_to_seconds(end_time_input)
        
        if (start_time_input and start_sec is None) or (end_time_input and end_sec is None):
            yield "Ошибка: Неверный формат времени. Используйте секунды или чч:мм:сс.", ""
            return

        audio_path = None
        audio_array = None
        sample_rate = None

        if isinstance(audio_input, str):
            original_path = audio_input
            ext = os.path.splitext(original_path)[1]
            safe_filename = f"{uuid.uuid4()}{ext}"
            temp_copy_path = os.path.join(str(TEMP_DIR), safe_filename)
            max_retries = 5
            retry_delay = 0.5
            for attempt in range(max_retries):
                try:
                    shutil.copy2(original_path, temp_copy_path)
                    audio_path = temp_copy_path
                    break
                except PermissionError as perm_err:
                    if attempt < max_retries - 1:
                        print(f"PermissionError при копировании (попытка {attempt + 1}/{max_retries}): {perm_err}")
                        time.sleep(retry_delay)
                    else:
                        print(f"Не удалось скопировать файл после {max_retries} попыток: {perm_err}")
                        audio_path = original_path
                except Exception as copy_err:
                    print(f"Не удалось скопировать файл: {copy_err}, используем оригинал")
                    audio_path = original_path
                    break
        elif isinstance(audio_input, tuple):
            sample_rate, audio_array = audio_input
        else:
            yield "Ошибка: Неверный формат аудио!", ""
            return

        if start_sec is not None or end_sec is not None:
            if audio_array is None or sample_rate is None:
                load_success = False
                for attempt in range(max_retries):
                    try:
                        audio_array, sample_rate = load_audio_use_ffmpeg(audio_path, resample=False)
                        load_success = True
                        break
                    except PermissionError as perm_err:
                        if attempt < max_retries - 1:
                            print(f"PermissionError при загрузке аудио (попытка {attempt + 1}/{max_retries}): {perm_err}")
                            time.sleep(retry_delay)
                        else:
                            yield f"Ошибка загрузки аудио после {max_retries} попыток: {perm_err}", ""
                            return
                    except Exception as exc:
                        yield f"Ошибка загрузки аудио: {exc}", ""
                        return
                if not load_success:
                    yield "Ошибка: Не удалось загрузить аудио", ""
                    return
            sliced_path, err = slice_audio_to_temp(audio_array, sample_rate, start_sec, end_sec)
            if err:
                yield f"Ошибка: {err}", ""
                return
            audio_path = sliced_path
        elif audio_array is not None and sample_rate is not None:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav", dir=str(TEMP_DIR))
            audio_path = temp_file.name
            temp_file.close()
            audio_data_int16 = (audio_array * 32768.0).astype(np.int16)
            sf.write(audio_path, audio_data_int16, sample_rate, subtype='PCM_16')
        
        streamer = TextIteratorStreamer(
            asr_model.processor.tokenizer, 
            skip_prompt=True, 
            skip_special_tokens=True
        )
        
        result_container = {"result": None, "error": None}
        
        def run_transcription():
            try:
                result_container["result"] = asr_model.transcribe(
                    audio_path=audio_path,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample,
                    repetition_penalty=repetition_penalty,
                    context_info=context_info if context_info and context_info.strip() else None,
                    streamer=streamer
                )
            except Exception as e:
                result_container["error"] = str(e)
                traceback.print_exc()
        
        start_time = time.time()
        transcription_thread = threading.Thread(target=run_transcription)
        transcription_thread.start()
        
        generated_text = ""
        token_count = 0
        was_stopped = False
        for new_text in streamer:
            # Проверяем флаг остановки
            if stop_generation_event.is_set():
                was_stopped = True
                break
            generated_text += new_text
            token_count += 1
            elapsed = time.time() - start_time
            formatted_text = generated_text.replace('},', '},\n')
            streaming_output = f"--- Распознавание... (токенов: {token_count}, время: {elapsed:.1f}с) ---\n{formatted_text}"
            yield streaming_output, "<div style='padding: 20px; text-align: center; color: #a0aec0;'>Генерация транскрипции... Аудио сегменты появятся после завершения.</div>"

        # Ожидаем завершения потока (если не остановлен)
        if not was_stopped and not stop_generation_event.is_set():
            transcription_thread.join()

        # Если была остановка - выводим сообщение
        if was_stopped or stop_generation_event.is_set():
            yield "--- Распознавание остановлено пользователем ---\n" + generated_text.replace('},', '},\n'), ""
            return
        
        if result_container["error"]:
            yield f"Ошибка транскрибации: {result_container['error']}", ""
            return
        
        result = result_container["result"]
        generation_time = time.time() - start_time
        
        input_tokens = result.get('input_tokens', {})
        speech_tokens = input_tokens.get('speech', 0)
        text_tokens = input_tokens.get('text', 0)
        padding_tokens = input_tokens.get('padding', 0)
        total_input = input_tokens.get('total', 0)
        
        raw_output = f"--- Результат распознавания ---\n"
        raw_output += f"Вход: {total_input} токенов (речь: {speech_tokens}, текст: {text_tokens}, паддинг: {padding_tokens})\n"
        raw_output += f"Выход: {token_count} токенов | Время: {generation_time:.2f}с\n"
        raw_output += f"---\n"
        formatted_raw_text = result['raw_text'].replace('},', '},\n')
        raw_output += formatted_raw_text
        
        audio_segments_html = ""
        segments = result['segments']
        
        if segments:
            num_segments = len(segments)
            audio_segments = extract_audio_segments(audio_path, segments)
            
            total_duration = sum(
                (seg.get('end_time', 0) - seg.get('start_time', 0)) 
                for seg in segments 
                if isinstance(seg.get('start_time'), (int, float)) and isinstance(seg.get('end_time'), (int, float))
            )
            approx_size_kb = total_duration * 4
            
            theme_css = """
            <style>
            .audio-segments-container {
                max-height: 600px;
                overflow-y: auto;
                padding: 10px;
            }
            
            .audio-segment {
                margin-bottom: 15px;
                padding: 15px;
                border: 2px solid #4a5568;
                border-radius: 8px;
                background-color: #2d3748;
                transition: all 0.3s ease;
            }
            
            .audio-segment:hover {
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
            }
            
            .segment-header {
                margin-bottom: 10px;
            }
            
            .segment-title {
                margin: 0;
                color: #e2e8f0;
                font-size: 16px;
                font-weight: 600;
            }
            
            .segment-meta {
                margin-top: 5px;
                font-size: 14px;
                color: #a0aec0;
            }
            
            .segment-content {
                margin-bottom: 10px;
                padding: 12px;
                background-color: #1a202c;
                border-radius: 6px;
                border-left: 4px solid #4299e1;
                color: #e2e8f0;
                line-height: 1.5;
            }
            
            .segment-audio {
                width: 100%;
                margin-top: 10px;
                border-radius: 4px;
            }
            
            .segment-warning {
                margin-top: 10px;
                padding: 10px;
                background-color: #744210;
                border-radius: 4px;
                border-left: 4px solid #d69e2e;
                color: #faf089;
                font-size: 13px;
            }
            
            .segments-title {
                color: #e2e8f0;
                margin-bottom: 10px;
            }
            
            .segments-description {
                color: #a0aec0;
                margin-bottom: 20px;
            }
            
            .size-badge {
                display: inline-block;
                background: linear-gradient(135deg, #4a5568, #2d3748);
                color: white;
                padding: 4px 10px;
                border-radius: 12px;
                font-size: 12px;
                margin-left: 10px;
            }
            </style>
            """
            
            audio_segments_html = theme_css
            audio_segments_html += f"<div class='audio-segments-container'>"
            
            format_info = "MP3 32kbps 16kHz моно" if HAS_PYDUB else "WAV 16kHz"
            audio_segments_html += f"<h3 class='segments-title'>Аудио сегменты ({num_segments} шт.)"
            audio_segments_html += f"<span class='size-badge'>~{approx_size_kb:.0f}КБ ({format_info})</span></h3>"
            audio_segments_html += "<p class='segments-description'>Нажмите на кнопку воспроизведения для прослушивания сегмента</p>"
            
            for i, (label, audio_src, error_msg) in enumerate(audio_segments):
                seg = segments[i] if i < len(segments) else {}
                start_time_seg = seg.get('start_time', 'N/A')
                end_time_seg = seg.get('end_time', 'N/A')
                speaker_id = seg.get('speaker_id', 'N/A')
                content = seg.get('text', '')
                
                start_str = f"{start_time_seg:.2f}" if isinstance(start_time_seg, (int, float)) else str(start_time_seg)
                end_str = f"{end_time_seg:.2f}" if isinstance(end_time_seg, (int, float)) else str(end_time_seg)
                
                audio_segments_html += f"""
                <div class='audio-segment'>
                    <div class='segment-header'>
                        <h4 class='segment-title'>Сегмент {i+1}</h4>
                        <div class='segment-meta'>
                            <strong>Время:</strong> [{start_str}с - {end_str}с] | 
                            <strong>Спикер:</strong> {speaker_id}
                        </div>
                    </div>
                    
                    <div class='segment-content'>
                        {content}
                    </div>
                """
                
                if audio_src:
                    audio_type = 'audio/mp3' if 'audio/mp3' in audio_src else 'audio/wav'
                    audio_segments_html += f"""
                    <audio controls class='segment-audio' preload='none'>
                        <source src='{audio_src}' type='{audio_type}'>
                        Ваш браузер не поддерживает аудио элемент.
                    </audio>
                    """
                elif error_msg:
                    audio_segments_html += f"""
                    <div class='segment-warning'>
                        <small>{error_msg}</small>
                    </div>
                    """
                else:
                    audio_segments_html += """
                    <div class='segment-warning'>
                        <small>Воспроизведение недоступно для этого сегмента</small>
                    </div>
                    """
                
                audio_segments_html += "</div>"
            
            audio_segments_html += "</div>"
        else:
            audio_segments_html = """
            <div style='padding: 20px; text-align: center; color: #a0aec0;'>
                <p>Аудио сегменты недоступны.</p>
                <p>Возможно, модель не вернула временные метки.</p>
            </div>
            """
        
        yield raw_output, audio_segments_html
        
    except Exception as e:
        print(f"Ошибка транскрибации: {e}")
        traceback.print_exc()
        yield f"Ошибка: {str(e)}", ""


def get_unique_speakers(segments) -> list:
    """Извлечение уникальных спикеров из сегментов."""
    speakers = set()
    for seg in segments:
        speaker = seg.get('speaker_id') if seg.get('speaker_id') is not None else seg.get('Speaker')
        if speaker is not None:
            speakers.add(speaker)
    return sorted(list(speakers))


def format_processed_text(segments, show_timestamps: bool, show_speakers: bool, show_descriptors: bool = False, selected_speakers: list = None) -> str:
    """Форматирование текста по спикерам с опциональными метками и фильтрацией дескрипторов."""
    if not segments:
        return ""
    
    lines = []
    for seg in segments:
        speaker = seg.get('speaker_id') if seg.get('speaker_id') is not None else seg.get('Speaker')
        if selected_speakers is not None and len(selected_speakers) > 0 and speaker not in selected_speakers:
            continue
        text = seg.get('text') or seg.get('Content', '')
        if isinstance(text, str):
            text = text.strip()
        if not text:
            continue
        
        if not show_descriptors:
            if re.match(r'^\[.+\]$', text):
                continue
            text = re.sub(r'\[.+?\]\s*', '', text)
        
        if not text.strip():
            continue
        
        prefix_parts = []
        if show_timestamps:
            start = seg.get('start_time') if seg.get('start_time') is not None else seg.get('Start', 0)
            end = seg.get('end_time') if seg.get('end_time') is not None else seg.get('End', 0)
            if isinstance(start, (int, float)) and isinstance(end, (int, float)):
                prefix_parts.append(f"[{start:.1f}s - {end:.1f}s]")
        
        if show_speakers:
            speaker = seg.get('speaker_id') if seg.get('speaker_id') is not None else seg.get('Speaker', 'N/A')
            prefix_parts.append(f"Спикер {speaker}:")
        
        if prefix_parts:
            lines.append(f"{' '.join(prefix_parts)} {text}")
        else:
            lines.append(text)
    
    return "\n".join(lines)


def create_gradio_interface():
    """Создание Gradio интерфейса."""
    
    custom_css = """
    #stop-btn {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%) !important;
        border: none !important;
        color: white !important;
    }
    #stop-btn:hover {
        background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%) !important;
    }
    .credits {
        text-align: left;
        padding: 10px;
        color: #a0aec0;
        font-size: 12px;
    }
    .credits a {
        color: #4299e1;
    }
    """
    
    dark_theme = gr.themes.Base(primary_hue="blue", neutral_hue="slate").set(
        body_background_fill="#1a202c",
        body_background_fill_dark="#1a202c",
        block_background_fill="#2d3748",
        block_background_fill_dark="#2d3748",
        block_border_color="#4a5568",
        block_border_color_dark="#4a5568",
        block_label_text_color="#e2e8f0",
        block_label_text_color_dark="#e2e8f0",
        block_title_text_color="#e2e8f0",
        block_title_text_color_dark="#e2e8f0",
        body_text_color="#e2e8f0",
        body_text_color_dark="#e2e8f0",
        body_text_color_subdued="#a0aec0",
        body_text_color_subdued_dark="#a0aec0",
        button_primary_background_fill="#4299e1",
        button_primary_background_fill_dark="#4299e1",
        button_primary_text_color="white",
        button_primary_text_color_dark="white",
        input_background_fill="#1a202c",
        input_background_fill_dark="#1a202c",
        input_border_color="#4a5568",
        input_border_color_dark="#4a5568",
    )
    
    with gr.Blocks(title="VibeVoice ASR - Распознавание речи", theme=dark_theme, css=custom_css) as demo:
        # queue() вызывается один раз перед launch() в main
        
        last_segments = gr.State([])
        
        gr.Markdown("# VibeVoice ASR - Распознавание речи в текст")
        
        gr.Markdown("Собрал [Nerual Dreaming](https://t.me/nerual_dreming) — основатель [ArtGeneration.me](https://artgeneration.me/), техноблогер и нейро-евангелист.")
        gr.Markdown("Канал [Нейро-Софт](https://t.me/neuroport) — репаки и портативки полезных нейросетей")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## Модель")
                
                model_dropdown = gr.Dropdown(
                    choices=[(name, path) for name, path in AVAILABLE_MODELS],
                    value=AVAILABLE_MODELS[0][1],
                    label="Выбор модели",
                    info="4-bit версия требует меньше VRAM, но работает медленнее"
                )
                
                use_4bit_checkbox = gr.Checkbox(
                    value=True,
                    label="Эмуляция 4-bit квантизации",
                    info="Экономит память GPU, но замедляет работу"
                )
                
                model_status = gr.Textbox(label="Статус", interactive=False, value="Модель не загружена", show_label=False)
                load_model_btn = gr.Button("Загрузить модель", variant="primary")
                
                context_info_input = gr.Textbox(
                    label="Контекст (опционально)",
                    placeholder="Введите ключевые слова, имена, термины...\nПример:\nИван Петров\nМашинное обучение\nOpenAI",
                    value="",
                    lines=4,
                    max_lines=8,
                    info="Помогает улучшить точность распознавания"
                )
                
                with gr.Accordion("Дополнительные параметры генерации", open=False):
                    max_tokens_slider = gr.Slider(
                        minimum=4096,
                        maximum=65536,
                        value=8192,
                        step=4096,
                        label="Максимум токенов"
                    )
                    
                    do_sample_checkbox = gr.Checkbox(
                        value=False,
                        label="Включить сэмплирование",
                        info="Случайная генерация вместо детерминированной"
                    )
                    
                    with gr.Column(visible=False) as sampling_params:
                        temperature_slider = gr.Slider(
                            minimum=0.0,
                            maximum=2.0,
                            value=0.0,
                            step=0.1,
                            label="Температура",
                            info="0 = жадный поиск, выше = больше случайности"
                        )
                        top_p_slider = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=1.0,
                            step=0.05,
                            label="Top-p (Nucleus Sampling)",
                            info="1.0 = без фильтрации"
                        )
                    
                    repetition_penalty_slider = gr.Slider(
                        minimum=1.0,
                        maximum=1.2,
                        value=1.0,
                        step=0.01,
                        label="Штраф за повторения",
                        info="1.0 = без штрафа, выше = меньше повторений"
                    )
                
                gr.Markdown("### Инструкция")
                gr.Markdown("""1. Выберите модель и нажмите "Загрузить модель"
2. Загрузите аудио/видео или запишите с микрофона
3. Контекст (опционально): добавьте ключевые слова
4. Нажмите "Распознать речь" и дождитесь результата
5. Просмотрите результаты во вкладках""")
            
            with gr.Column(scale=2):
                gr.Markdown("## Входные данные")
                
                active_input_tab = gr.State(0)
                
                with gr.Tabs() as input_tabs:
                    with gr.TabItem("Аудио", id=0) as audio_tab:
                        audio_input = gr.Audio(
                            label="Аудио файл",
                            sources=["upload"],
                            type="filepath",
                            interactive=True
                        )
                    
                    with gr.TabItem("Видео", id=1) as video_tab:
                        video_input = gr.Video(
                            label="Видео файл",
                            sources=["upload"],
                            interactive=True
                        )
                    
                    with gr.TabItem("Микрофон", id=2) as mic_tab:
                        mic_input = gr.Audio(
                            label="Запишите с микрофона",
                            sources=["microphone"],
                            type="filepath",
                            interactive=True
                        )
                
                audio_tab.select(fn=lambda: 0, inputs=[], outputs=[active_input_tab])
                video_tab.select(fn=lambda: 1, inputs=[], outputs=[active_input_tab])
                mic_tab.select(fn=lambda: 2, inputs=[], outputs=[active_input_tab])
                
                with gr.Row():
                    transcribe_button = gr.Button("Распознать речь", variant="primary", size="lg", visible=True)
                    stop_button = gr.Button("Стоп", variant="stop", size="lg", visible=False)
                
                gr.Markdown("## Результаты")
                
                with gr.Tabs():
                    with gr.TabItem("Сырой текст"):
                        raw_output = gr.Textbox(
                            label="Распознанный текст (JSON)",
                            lines=8,
                            max_lines=20,
                            interactive=False
                        )
                    
                    with gr.TabItem("Обработанный текст"):
                        with gr.Row():
                            show_timestamps_checkbox = gr.Checkbox(
                                value=False,
                                label="Временные метки"
                            )
                            show_speakers_checkbox = gr.Checkbox(
                                value=False,
                                label="Спикеры"
                            )
                            show_descriptors_checkbox = gr.Checkbox(
                                value=False,
                                label="Показать дескрипторы"
                            )
                        speaker_filter = gr.CheckboxGroup(
                            choices=[],
                            value=[],
                            label="Фильтр по спикерам",
                            visible=False
                        )
                        processed_output = gr.Textbox(
                            label="Текст по спикерам",
                            lines=15,
                            max_lines=30,
                            interactive=True,
                            placeholder="Генерация транскрипции... Обработанный текст появится после завершения."
                        )
                        copy_btn = gr.Button("Копировать текст", variant="secondary", size="sm")
                    
                    with gr.TabItem("Аудио сегменты"):
                        audio_segments_output = gr.HTML(
                            label="Прослушайте отдельные сегменты"
                        )
        
        
        
        def reset_stop_flag():
            """Сброс флага остановки перед началом новой транскрибации."""
            stop_generation_event.clear()

        def set_stop_flag():
            """Установка флага остановки."""
            stop_generation_event.set()
        
        def load_model_handler(model_path, use_4bit):
            return initialize_model(model_path, use_4bit)
        
        def on_model_change(model_path):
            is_4bit_model = "4bit" in model_path.lower() or "4-bit" in model_path.lower()
            return gr.update(value=False, interactive=not is_4bit_model)
        
        def update_processed_text(segments, show_timestamps, show_speakers, show_descriptors, selected_speakers):
            speaker_ids = None
            if selected_speakers:
                speaker_ids = []
                for s in selected_speakers:
                    try:
                        speaker_id = int(s.replace("Спикер ", ""))
                        speaker_ids.append(speaker_id)
                    except (ValueError, AttributeError):
                        pass
            return format_processed_text(segments, show_timestamps, show_speakers, show_descriptors, speaker_ids)
        
        do_sample_checkbox.change(
            fn=lambda x: gr.update(visible=x),
            inputs=[do_sample_checkbox],
            outputs=[sampling_params]
        )
        
        model_dropdown.change(
            fn=on_model_change,
            inputs=[model_dropdown],
            outputs=[use_4bit_checkbox]
        )
        
        load_model_btn.click(
            fn=load_model_handler,
            inputs=[model_dropdown, use_4bit_checkbox],
            outputs=[model_status]
        )
        
        def transcribe_wrapper(
            audio_file, video_file, mic_file, active_tab,
            max_tokens, temperature, top_p, do_sample, rep_penalty, context,
            model_path, use_4bit, current_segments,
            show_timestamps, show_speakers, show_descriptors
        ):
            global asr_model
            
            if active_tab == 0:
                selected_input = audio_file
            elif active_tab == 1:
                selected_input = video_file
            else:
                selected_input = mic_file
            
            segments_list = []
            raw_text = ""
            audio_html = ""
            model_status_text = ""
            
            if asr_model is None:
                model_status_text = "Загрузка модели..."
            else:
                model_status_text = f"Модель загружена: {model_path.split('/')[-1]}"
            
            for raw_text, audio_html in transcribe_audio(
                selected_input if active_tab == 0 else None,
                selected_input if active_tab == 1 else None,
                selected_input if active_tab == 2 else None,
                "",
                "", "",
                max_tokens, temperature, top_p, do_sample, rep_penalty, context,
                model_path, use_4bit
            ):
                if asr_model is not None:
                    model_status_text = f"Модель загружена: {model_path.split('/')[-1]}"
                yield raw_text, audio_html, segments_list, "", model_status_text, gr.update()
            
            try:
                start_idx = raw_text.find('[')
                end_idx = raw_text.rfind(']')
                if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                    json_str = raw_text[start_idx:end_idx+1]
                    segments_list = json.loads(json_str)
            except Exception as e:
                print(f"Ошибка парсинга сегментов: {e}")
            
            if asr_model is not None:
                model_status_text = f"Модель загружена: {model_path.split('/')[-1]}"
            
            unique_speakers = get_unique_speakers(segments_list)
            speaker_filter_update = gr.update(
                choices=[f"Спикер {s}" for s in unique_speakers],
                value=[f"Спикер {s}" for s in unique_speakers],
                visible=len(unique_speakers) > 1
            )
            
            processed = format_processed_text(segments_list, show_timestamps, show_speakers, show_descriptors)
            yield raw_text, audio_html, segments_list, processed, model_status_text, speaker_filter_update
        
        def start_transcription():
            reset_stop_flag()
            return gr.update(visible=False), gr.update(visible=True)
        
        def finish_transcription():
            return gr.update(visible=True), gr.update(visible=False)
        
        def stop_transcription():
            """Остановка транскрибации - устанавливает флаг и возвращает кнопки в исходное состояние."""
            set_stop_flag()
            return gr.update(visible=True), gr.update(visible=False)
        
        transcribe_event = transcribe_button.click(
            fn=start_transcription,
            inputs=[],
            outputs=[transcribe_button, stop_button],
            queue=False
        ).then(
            fn=transcribe_wrapper,
            inputs=[
                audio_input, video_input, mic_input, active_input_tab,
                max_tokens_slider, temperature_slider, top_p_slider,
                do_sample_checkbox, repetition_penalty_slider, context_info_input,
                model_dropdown, use_4bit_checkbox, last_segments,
                show_timestamps_checkbox, show_speakers_checkbox, show_descriptors_checkbox
            ],
            outputs=[raw_output, audio_segments_output, last_segments, processed_output, model_status, speaker_filter]
        ).then(
            fn=finish_transcription,
            inputs=[],
            outputs=[transcribe_button, stop_button],
            queue=False
        )
        
        stop_button.click(
            fn=stop_transcription,
            inputs=[],
            outputs=[transcribe_button, stop_button],
            queue=False,
            cancels=[transcribe_event]  # Отменяет запущенную транскрибацию в очереди
        )
        
        show_timestamps_checkbox.change(
            fn=update_processed_text,
            inputs=[last_segments, show_timestamps_checkbox, show_speakers_checkbox, show_descriptors_checkbox, speaker_filter],
            outputs=[processed_output]
        )
        
        show_speakers_checkbox.change(
            fn=update_processed_text,
            inputs=[last_segments, show_timestamps_checkbox, show_speakers_checkbox, show_descriptors_checkbox, speaker_filter],
            outputs=[processed_output]
        )
        
        show_descriptors_checkbox.change(
            fn=update_processed_text,
            inputs=[last_segments, show_timestamps_checkbox, show_speakers_checkbox, show_descriptors_checkbox, speaker_filter],
            outputs=[processed_output]
        )
        
        speaker_filter.change(
            fn=update_processed_text,
            inputs=[last_segments, show_timestamps_checkbox, show_speakers_checkbox, show_descriptors_checkbox, speaker_filter],
            outputs=[processed_output]
        )
        
        def copy_text(text):
            return gr.update(value="Скопировано!")
        
        def reset_copy_btn():
            return gr.update(value="Копировать текст")
        
        copy_btn.click(
            fn=copy_text,
            inputs=[processed_output],
            outputs=[copy_btn],
            js="(text) => { navigator.clipboard.writeText(text); return text; }"
        ).then(
            fn=reset_copy_btn,
            inputs=[],
            outputs=[copy_btn],
            js="() => new Promise(resolve => setTimeout(resolve, 1500))"
        )
    
    return demo


if __name__ == "__main__":
    print("=" * 60)
    print(f"{APP_NAME} v{APP_VERSION}")
    print("=" * 60)
    print()
    
    device = get_device()
    print(f"Устройство: {device.upper()}")
    
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    attn_impl = get_attn_implementation()
    print(f"Attention: {attn_impl}")
    
    print()
    print(f"Папка вывода: {OUTPUT_DIR}")
    print(f"Временная папка: {TEMP_DIR}")
    print()
    print("Запуск веб-интерфейса...")
    print()
    
    demo = create_gradio_interface()
    
    allowed_paths = [str(SCRIPT_DIR), str(TEMP_DIR), str(OUTPUT_DIR)]
    if sys.platform == "win32":
        import string
        for drive in string.ascii_uppercase:
            drive_path = f"{drive}:\\"
            if os.path.exists(drive_path):
                allowed_paths.append(drive_path)
        gr.set_static_paths(paths=allowed_paths)
    
    demo.queue(default_concurrency_limit=2).launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
        inbrowser=True,
        allowed_paths=allowed_paths
    )
