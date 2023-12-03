mod ffmpeg_decoder;
mod model;
mod transcript;
mod utils;
mod whisper;

use anyhow::Result;
pub use model::{Model, Size};
use std::path::Path;
pub use transcript::{Transcript, Utternace};
pub use whisper::{Language, Whisper};

pub async fn transcribe_audio<P: AsRef<Path>>(
    audio: P,
    model: P,
    prompt: Option<&str>,
    response_format: Option<&str>,
    temperature: Option<f32>,
    lang: Option<&str>,
) -> Result<String> {
    let mut whisper = Whisper::from_model_path(model, Some(Language::Auto)).await;
    let transcript = whisper.transcribe(audio, false, false, prompt)?;

    let response_format = response_format.unwrap_or("text");
    match response_format {
        "srt" => Ok(transcript.as_srt()),
        "vtt" => Ok(transcript.as_vtt()),
        _ => Ok(transcript.as_text()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    #[tokio::test]
    async fn test_transcribe_audio() {
        let GGML_METAL_PATH_RESOURCES = "/Users/jiahua/rust_code/whisper.cpp";
        env::set_var("GGML_METAL_PATH_RESOURCES", GGML_METAL_PATH_RESOURCES);
        let audio = "/Users/jiahua/rust_code/whisper-rs/examples/full_usage/2830-3980-0043.wav";
        let model = "/Users/jiahua/rust_code/whisper.cpp/models/ggml-base.en.bin";
        let transcript = transcribe_audio(audio, model, None, None, None, None)
            .await
            .unwrap();
        println!("{}", transcript);
    }
}
