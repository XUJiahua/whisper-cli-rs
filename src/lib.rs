use std::path::Path;

use anyhow::Result;

pub use model::{Model, Size};
pub use transcript::{Transcript, Utternace};
pub use whisper::{Language, Whisper};

mod ffmpeg_decoder;
mod model;
mod transcript;
mod utils;
mod whisper;

pub async fn transcribe_audio<P: AsRef<Path>, Q: AsRef<Path>, F>(
    audio: P,
    model: Q,
    prompt: Option<&str>,
    response_format: Option<&str>,
    _temperature: Option<f32>,
    _lang: Option<&str>,
    progress: F,
) -> Result<String>
where
    F: FnMut(i32) + 'static,
{
    let whisper = Whisper::from_model_path(model, Some(Language::Auto)).await;
    let transcript = whisper.transcribe(audio, false, false, prompt, progress)?;

    let response_format = response_format.unwrap_or("text");
    match response_format {
        "srt" => Ok(transcript.as_srt()),
        "vtt" => Ok(transcript.as_vtt()),
        _ => Ok(transcript.as_text()),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, RwLock};
    use std::{env, thread};

    use super::*;

    #[tokio::test]
    async fn test_transcribe_audio() {
        // fix: how to collect progress? 目前为止，这种方式可行
        let latest_progress = Arc::new(RwLock::new(0));
        let latest_progress_in_callback = Arc::clone(&latest_progress);

        thread::spawn(move || loop {
            let latest_progress = latest_progress.read().unwrap();
            println!("progress: {}", *latest_progress);
            thread::sleep(std::time::Duration::from_secs(1));
        });
        let GGML_METAL_PATH_RESOURCES = "/Users/jiahua/rust_code/whisper.cpp";
        env::set_var("GGML_METAL_PATH_RESOURCES", GGML_METAL_PATH_RESOURCES);
        let audio = "/Users/jiahua/rust_code/whisper-rs/examples/full_usage/2830-3980-0043.wav";
        let model = "/Users/jiahua/rust_code/whisper.cpp/models/ggml-base.en.bin";
        let transcript = transcribe_audio(audio, model, None, None, None, None, move |i| {
            let mut latest_progress = latest_progress_in_callback.write().unwrap();
            *latest_progress = i;
        })
        .await
        .unwrap();

        println!("{}", transcript);
    }
}
