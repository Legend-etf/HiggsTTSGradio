import os
import glob
import gradio as gr
import torch
import soundfile as sf
import torchaudio
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "multimodalart/higgs-audio-v3-tts-4b-transformers"
VOICE_DIR = "./voices" 
OUTPUT_FILE = "output.wav"

CONTROL_TOKENS = [

    "<|emotion:elation|>",
    "<|emotion:amusement|>",
    "<|emotion:enthusiasm|>",
    "<|emotion:determination|>",
    "<|emotion:pride|>",
    "<|emotion:contentment|>",
    "<|emotion:affection|>",
    "<|emotion:relief|>",
    "<|emotion:contemplation|>",
    "<|emotion:confusion|>",
    "<|emotion:surprise|>",
    "<|emotion:awe|>",
    "<|emotion:longing|>",
    "<|emotion:arousal|>",
    "<|emotion:anger|>",
    "<|emotion:fear|>",
    "<|emotion:disgust|>",
    "<|emotion:bitterness|>",
    "<|emotion:sadness|>",
    "<|emotion:shame|>",
    "<|emotion:helplessness|>",
    "<|style:singing|>",
    "<|style:shouting|>",
    "<|style:whispering|>",
    "<|sfx:cough|>",
    "<|sfx:laughter|>",
    "<|sfx:crying|>",
    "<|sfx:screaming|>",
    "<|sfx:burping|>",
    "<|sfx:humming|>",
    "<|sfx:sigh|>",
    "<|sfx:sniff|>",
    "<|sfx:sneeze|>",
    "<|prosody:speed_very_slow|>",
    "<|prosody:speed_slow|>",
    "<|prosody:speed_fast|>",
    "<|prosody:speed_very_fast|>",
    "<|prosody:pitch_low|>",
    "<|prosody:pitch_high|>",
    "<|prosody:pause|>",
    "<|prosody:long_pause|>",
    "<|prosody:expressive_high|>",
    "<|prosody:expressive_low|>",
]

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto",
).eval()

def get_voice_list():
    exts = ("*.wav", "*.mp3", "*.flac", "*.ogg", "*.m4a")
    files = []

    for ext in exts:
        files.extend(glob.glob(os.path.join(VOICE_DIR, ext)))

    files.sort()

    if not files:
        return []

    return files

def append_token(current_text, token):
    current_text = current_text or ""

    if current_text.endswith(" ") or current_text == "":
        return current_text + token

    return current_text + " " + token

def synthesize(text, voice_path, voice_transcript):
    if not text.strip():
        raise gr.Error("Please enter text.")

    if not voice_path:
        raise gr.Error("Please select a reference voice.")
    
    if not voice_transcript.strip():
        voice_transcript = None
        print("No reference transcript provided.")
        
    ref, sr = torchaudio.load(voice_path)

    with torch.no_grad():
        wav = model.generate_speech(
            text,
            tokenizer,
            reference_audio=ref,
            reference_text=voice_transcript,
            reference_sample_rate=sr,
            max_new_tokens=2024,
            temperature=1.0
        )

    if torch.is_tensor(wav):
        wav = wav.detach().cpu().numpy()

    sf.write(
        OUTPUT_FILE,
        wav,
        model.config.sample_rate,
    )

    return OUTPUT_FILE

with gr.Blocks(
    head="""
    <script>
    window.insertControlToken = function(token) {
        const textarea = document.querySelector(
            "#synth_textbox textarea"
        );

        if (!textarea) {
            console.log("Textarea not found");
            return;
        }

        textarea.focus();

        const start = textarea.selectionStart;
        const end = textarea.selectionEnd;

        textarea.setRangeText(
            token,
            start,
            end,
            "end"
        );

        textarea.dispatchEvent(
            new Event("input", { bubbles: true })
        );
    };
    </script>
    """
    ) as demo:

    gr.Markdown("# Higgs Audio v3 TTS")

    with gr.Row():

        voice_dropdown = gr.Dropdown(
            choices=get_voice_list(),
            label="Reference Voice",
            value=get_voice_list()[0] if get_voice_list() else None,
        )
        
        voice_transcript = gr.Textbox(
            label="Reference Transcript",
            placeholder="Optional: Enter transcript of the reference voice for better results...",
            elem_id="voice_transcript",
        )

        refresh_button = gr.Button("🔄 Refresh Voices")

    text_input = gr.Textbox(
        label="Text",
        lines=8,
        elem_id="synth_textbox",
        placeholder="Enter text to synthesize...",
    )
    
    gr.Markdown("## Control Tokens:")

    with gr.Row():
        for token in CONTROL_TOKENS:
            gr.Button(token, size="sm").click(
                fn=None,
                js=f"""
                () => {{
                    window.insertControlToken({token!r});
                }}
                """
            )       

   

    generate_button = gr.Button(
        "Generate",
        variant="primary",
    )

    output_audio = gr.Audio(
        label="Generated Audio",
        value=None,
        autoplay=False,
    )

    refresh_button.click(
        fn=lambda: gr.update(choices=get_voice_list()),
        outputs=voice_dropdown,
    )

    generate_button.click(
        fn=synthesize,
        inputs=[
            text_input,
            voice_dropdown,
            voice_transcript,
        ],
        outputs=output_audio,
    )

demo.launch()