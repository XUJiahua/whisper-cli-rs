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
    language: Option<String>,
) -> Result<String> {
    let mut whisper = Whisper::from_model_path(model, None).await;
    let transcript = whisper.transcribe(audio, false, false)?;

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
        let audio = "/Users/jiahua/.lwt/data/3048d90c-ed14-4b97-a3cb-49d50ee37d5d/db81a94f-db0b-4829-9b12-c02283812615/Running LLaMA 7B and 13B on a 64GB M2 MacBook Pro with llama.cpp _ Simon Willison’s TILs_20231201080824.mp3";
        let model = "/Users/jiahua/rust_code/whisper.cpp/models/ggml-base.en.bin";
        let transcript = transcribe_audio(audio, model, None, None, None, None)
            .await
            .unwrap();
        println!("{}", transcript);
    }
}
