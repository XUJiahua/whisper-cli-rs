#![warn(clippy::all, clippy::pedantic, clippy::nursery)]

use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::{convert::Infallible, net::SocketAddr};

use clap::Parser;
use hyper::service::{make_service_fn, service_fn};
use hyper::{header::CONTENT_TYPE, Body, Request, Response, Server, StatusCode};
use multer::Multipart;
use serde::Serialize;
use serde_json::to_string;

use whisper_cli::{Language, Model, Size, Whisper};

use crate::utils::write_to;

mod utils;

#[derive(Serialize)]
struct TranscriptionResponse {
    text: String,
}

#[derive(Parser)]
struct Opts {
    #[clap(subcommand)]
    subcmd: SubCommand,
}

#[derive(Parser)]
enum SubCommand {
    #[command(about = "Start the transcription server.")]
    Serve {
        /// Port to listen on
        #[clap(short, long, default_value = "8000")]
        port: u16,
        /// Path to the Whisper model
        #[clap(short, long)]
        model_path: String,
    },
    #[command(about = "Transcribe a given audio file.")]
    Transcribe(TranscribeArgs),
}

#[derive(Parser)]
struct TranscribeArgs {
    /// Name of the Whisper model to use
    #[clap(short, long, default_value = "medium")]
    model: Size,

    /// Language spoken in the audio. Attempts to auto-detect by default.
    #[clap(short, long)]
    lang: Option<Language>,

    /// Path to the audio file to transcribe
    #[clap(name = "AUDIO")]
    audio: String,

    /// Toggle translation
    #[clap(short, long, default_value = "false")]
    translate: bool,

    /// Generate timestamps for each word
    #[clap(short, long, default_value = "false")]
    karaoke: bool,

    /// Write transcription results to .txt, .vtt, and .srt files.
    #[clap(short, long, default_value = "false")]
    write: bool,
}

#[tokio::main]
async fn main() {
    let opts = Opts::parse();
    match opts.subcmd {
        SubCommand::Serve { port, model_path } => {
            let model_path = Path::new(&model_path);
            start_server(port, &model_path).await;
        }
        SubCommand::Transcribe(args) => transcribe_audio(args).await,
    }
}

const JSON_FORMAT: &str = "json";
const TEXT_FORMAT: &str = "text";
const SRT_FORMAT: &str = "srt";
const VJSON_FORMAT: &str = "verbose_json";
const VTT_FORMAT: &str = "vtt";

const CONTENT_TYPE_JSON: &str = "application/json; charset=utf-8";
const CONTENT_TYPE_TEXT: &str = "text/plain; charset=utf-8";

async fn start_server(port: u16, model_path: &Path) {
    // load model
    let whisper = Arc::new(Mutex::new(
        Whisper::from_model_path(model_path, Some(Language::Auto)).await,
    ));

    let make_svc = make_service_fn(move |_conn| {
        let whisper_clone = whisper.clone();
        async move {
            Ok::<_, Infallible>(service_fn(move |req| {
                handle_transcription(req, whisper_clone.clone())
            }))
        }
    });

    // start listening
    let addr = SocketAddr::from(([127, 0, 0, 1], port));
    let server = Server::bind(&addr).serve(make_svc);

    println!("🏃‍♀️ Server running at: {}", addr);
    if let Err(e) = server.await {
        eprintln!("server error: {}", e);
    }
}

// A handler for incoming requests.
async fn handle_transcription(
    req: Request<Body>,
    whisper: Arc<Mutex<Whisper>>,
) -> Result<Response<Body>, Infallible> {
    // Check if the request is a preflight request (OPTIONS method)
    if req.method() == hyper::Method::OPTIONS && req.uri().path() == "/v1/audio/transcriptions" {
        let res = Response::builder()
            .status(StatusCode::OK)
            .header("Access-Control-Allow-Origin", "*")
            .header("Access-Control-Allow-Methods", "POST, OPTIONS")
            .header("Access-Control-Allow-Headers", "Content-Type")
            .body(Body::empty())
            .unwrap();
        return Ok(res);
    }
    // Extract the `multipart/form-data` boundary from the headers.
    let boundary = req
        .headers()
        .get(CONTENT_TYPE)
        .and_then(|ct| ct.to_str().ok())
        .and_then(|ct| multer::parse_boundary(ct).ok());

    // Send `BAD_REQUEST` status if the content-type is not multipart/form-data.
    if boundary.is_none() {
        return Ok(Response::builder()
            .status(StatusCode::BAD_REQUEST)
            .header("Access-Control-Allow-Origin", "*") // Add this for CORS
            .body(Body::from("BAD REQUEST"))
            .unwrap());
    }

    // Process the multipart e.g. you can store them in files.
    let transcription_request = process_multipart(req.into_body(), boundary.unwrap()).await;

    if let Err(err) = transcription_request {
        return Ok(Response::builder()
            .status(StatusCode::INTERNAL_SERVER_ERROR)
            .header("Access-Control-Allow-Origin", "*") // Add this for CORS
            .body(Body::from(format!("INTERNAL SERVER ERROR: {}", err)))
            .unwrap());
    }

    if let Ok((trans_req, fields)) = transcription_request {
        let audio = Path::new(trans_req.as_str());
        let prompt = fields.get("prompt");
        let transcript = {
            let mut whisper_guard = whisper.lock().unwrap();
            whisper_guard
                .transcribe(audio, false, false, prompt.map(|s| s.as_str()), |p| {})
                .unwrap()
        };
        println!("time: {:?}", transcript.processing_time);
        println!("fields: {:?}", fields);

        let lang = fields.get("language");
        let response_format = fields
            .get("response_format")
            .map(|s| s.as_str())
            .unwrap_or(JSON_FORMAT);
        let mut is_json = false;
        let response_body = match response_format {
            SRT_FORMAT => transcript.as_srt(),
            VTT_FORMAT => transcript.as_vtt(),
            TEXT_FORMAT => transcript.as_text(),
            &_ => {
                is_json = true;
                let transcript_text = transcript.as_text();
                let response: TranscriptionResponse = TranscriptionResponse {
                    text: transcript_text,
                };
                let json_response = to_string(&response).expect("Failed to serialize to JSON");

                json_response
            }
        };

        let content_type = if is_json {
            CONTENT_TYPE_JSON
        } else {
            CONTENT_TYPE_TEXT
        };
        let response = Response::builder()
            .header("Access-Control-Allow-Origin", "*") // Add this for CORS
            .header("content-type", content_type)
            .body(Body::from(response_body))
            .unwrap();

        return Ok(response);
    }

    Ok(Response::new(Body::from("Success")))
}

// Process the request body as multipart/form-data.
async fn process_multipart(
    body: Body,
    boundary: String,
) -> multer::Result<(String, HashMap<String, String>)> {
    // Create a Multipart instance from the request body.
    let mut multipart = Multipart::new(body, boundary);
    let mut file_path = String::new();
    let mut other_fields = HashMap::new();

    // Iterate over the fields, `next_field` method will return the next field if
    // available.
    while let Some(mut field) = multipart.next_field().await? {
        if field.name() == Some("file") {
            // Get the field name.
            let name = field.name();

            // Get the field's filename if provided in "Content-Disposition" header.
            let file_name = field.file_name();

            // Get the "Content-Type" header as `mime::Mime` type.
            let content_type = field.content_type();

            println!(
                "Name: {:?}, FileName: {:?}, Content-Type: {:?}",
                name, file_name, content_type
            );
            // Process the field data chunks e.g. store them in a file.
            let mut bytes_len = 0;
            let mut audio_data = Vec::new();
            while let Some(field_chunk) = field.chunk().await? {
                audio_data.extend_from_slice(&field_chunk);
                bytes_len += field_chunk.len();
            }
            println!("Bytes Length: {:?}", bytes_len);
            let file_name_str: &str = field.file_name().as_ref().unwrap_or(&"audio.wav");
            file_path = format!("/tmp/{}", file_name_str); // Adjust as necessary
            fs::write(&file_path, audio_data).expect("Failed to write to file");
            println!("Write the file to {}", file_path);
        } else {
            let mut value = String::new();
            while let Some(field_chunk) = field.chunk().await? {
                value.push_str(&String::from_utf8_lossy(&field_chunk));
            }
            if let Some(field_name) = field.name() {
                other_fields.insert(field_name.to_string(), value);
            }
        }
    }

    Ok((file_path, other_fields))
}

async fn transcribe_audio(mut args: TranscribeArgs) {
    let audio = Path::new(&args.audio);
    let file_name = audio.file_name().unwrap().to_str().unwrap();

    assert!(audio.exists(), "The provided audio file does not exist.");

    if args.model.is_english_only() && (args.lang == Some(Language::Auto) || args.lang.is_none()) {
        args.lang = Some(Language::English);
    }

    assert!(
        !args.model.is_english_only() || args.lang == Some(Language::English),
        "The selected model only supports English."
    );

    let mut whisper = Whisper::new(Model::new(args.model), args.lang).await;
    let transcript = whisper
        .transcribe(audio, args.translate, args.karaoke, None, |p| {})
        .unwrap();
    println!("time: {:?}", transcript.processing_time);

    if args.write {
        write_to(
            audio.with_file_name(format!("{file_name}.txt")),
            &transcript.as_text(),
        );
        write_to(
            audio.with_file_name(format!("{file_name}.vtt")),
            &transcript.as_vtt(),
        );
        write_to(
            audio.with_file_name(format!("{file_name}.srt")),
            &transcript.as_srt(),
        );
    } else {
        println!("");
        println!("🔊 {}", transcript.as_text());
    }
}
