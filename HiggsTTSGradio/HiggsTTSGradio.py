import gradio as gr
import torch
import soundfile as sf
import torchaudio
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
BASE_DIR = Path(__file__).resolve().parent
VOICE_DIR = BASE_DIR / "voices"
MODEL_PATH = "multimodalart/higgs-audio-v3-tts-4b-transformers" #https://huggingface.co/multimodalart/higgs-audio-v3-tts-4b-transformers
OUTPUT_FILE = "output.wav"
TEXT_INPUTS = []
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
    "<|prosody:expressive_low|>"
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
    dtype=torch.bfloat16,
    device_map="cuda:0"
).eval()

def get_voice_list():
    exts = ("*.wav", "*.mp3", "*.flac", "*.ogg", "*.m4a")
    files = []

    for ext in exts:
        files.extend(
            str(p.relative_to(BASE_DIR))
            for p in VOICE_DIR.glob(ext)
        )
    if not files:
        return []

    return sorted(files)

def load_transcript(voice_path):
    if not voice_path:
        return ""

    voice_path = BASE_DIR / voice_path

    txt_path = voice_path.with_suffix(".txt")

    if txt_path.exists():
        return txt_path.read_text(encoding="utf-8")
    
def append_token(current_text, token):
    current_text = current_text or ""

    if current_text.endswith(" ") or current_text == "":
        return current_text + token

    return current_text + " " + token

def synthesize(*args):
    *texts, voice_path, voice_transcript, seed = args

    pieces = []
    voice_path = BASE_DIR / voice_path
    ref, sr = torchaudio.load(voice_path)
    codes = model._encode_reference(ref, sr)
    texts = [t 
            for t in texts
                if t.strip()
            ]

    if seed == -1:
        seed = torch.randint(0, 10000, (1,)).item()

    if not texts[0].strip():
        raise gr.Error("Please enter text for the first input.")

    if not voice_path:
        raise gr.Error("Please select a reference voice.")
    
    if not voice_transcript.strip():
        voice_transcript = None
        print("No reference transcript provided.")
        
    for text in texts:
        torch.manual_seed(seed)
        with torch.inference_mode():
            wav = model.generate_speech(
                text,
                tokenizer,
                reference_codes=codes,
                reference_text=voice_transcript,
                reference_sample_rate=sr,
                max_new_tokens=1024,
                temperature=1.0
            )
        pieces.append(wav)
    final = torch.cat(pieces)

    if torch.is_tensor(final):
        final = final.detach().cpu().numpy()

    sf.write(
        OUTPUT_FILE,
        final,
        model.config.sample_rate,
    )
    return OUTPUT_FILE

with gr.Blocks(
    head="""
    <script>
    window.activeTextbox = null;

    document.addEventListener("focusin", (event) => {
        if (event.target.tagName.toUpperCase() === "TEXTAREA") {
            window.activeTextbox = event.target;
        }
    });

    window.insertControlToken = function(token) {

        const textarea = window.activeTextbox;

        if (!textarea || !document.contains(textarea)) {
            console.log("No active textbox");
            return;
        }

        textarea.focus();

        textarea.setRangeText(
            token,
            textarea.selectionStart,
            textarea.selectionEnd,
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

        voice_dropdown.change(
            fn=load_transcript,
            inputs=voice_dropdown,
            outputs=voice_transcript,
        )

        refresh_button = gr.Button("🔄 Refresh Voices")
        
    with gr.Row(variant="compact"):
        seed = gr.Slider(
            label="Seed (set to -1 to randomise)",
            minimum=-1,
            maximum=10000,
            value=-1,
            step=1
        )

    for i in range(6):
        text_input = gr.Textbox(
            label=f"Text {i+1}",
            lines=2,
            elem_id=f"synth_textbox_{i}",
            placeholder="Enter text to synthesize...",
        )
        TEXT_INPUTS.append(text_input)

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

    cancel_button = gr.Button(
        "Cancel",
        variant="secondary",
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

    generate_event = generate_button.click(
        fn=synthesize,
        inputs=[
            *TEXT_INPUTS,
            voice_dropdown,
            voice_transcript,
            seed
        ],
        outputs=output_audio,
    )

    cancel_button.click(
        fn=lambda: None,
        cancels=generate_event,
    )

demo.launch()