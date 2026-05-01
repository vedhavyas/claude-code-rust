//! AskUserQuestion sequential sequencing. Mirrors upstream's
//! `agent-sdk/src/bridge/user_interaction.ts` (323 LoC) — focused on
//! the multi-question loop + answer-payload assembly.
//!
//! The actual loop driver lives in `forge_sdk_worker::run_ask_user_question`
//! because it needs the worker's `pending` map + event_tx + the
//! oneshot machinery. This module owns the pure parsing helpers.

use serde_json::{Map, Value};

use crate::agent::types::{
    QuestionAnnotation, QuestionOption as TuiQuestionOption, QuestionPrompt, QuestionRequest,
    ToolCall,
};

pub const ASK_USER_QUESTION_TOOL_NAME: &str = "AskUserQuestion";

#[derive(Debug, Clone)]
pub struct AskUserQuestionOption {
    pub label: String,
    pub description: String,
    pub preview: Option<String>,
}

#[derive(Debug, Clone)]
pub struct AskUserQuestionPrompt {
    pub question: String,
    pub header: String,
    pub multi_select: bool,
    pub options: Vec<AskUserQuestionOption>,
}

/// Mirrors upstream's `parseAskUserQuestionPrompts`. Walks the raw
/// `input.questions` JSON array, dropping malformed entries and
/// requiring at least 2 options per question (matches upstream's
/// validity rule).
#[must_use]
pub fn parse_ask_user_question_prompts(input: &Value) -> Vec<AskUserQuestionPrompt> {
    let Some(questions) = input.as_object().and_then(|r| r.get("questions")).and_then(Value::as_array) else {
        return Vec::new();
    };
    let mut prompts: Vec<AskUserQuestionPrompt> = Vec::new();
    for raw in questions {
        let Some(q) = raw.as_object() else { continue };
        let question = q.get("question").and_then(Value::as_str).unwrap_or("").trim().to_owned();
        if question.is_empty() {
            continue;
        }
        let header_raw = q.get("header").and_then(Value::as_str).unwrap_or("").trim().to_owned();
        let header = if header_raw.is_empty() { format!("Q{}", prompts.len() + 1) } else { header_raw };
        let multi_select = q
            .get("multiSelect")
            .or_else(|| q.get("multi_select"))
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let mut options: Vec<AskUserQuestionOption> = Vec::new();
        if let Some(opts) = q.get("options").and_then(Value::as_array) {
            for raw_opt in opts {
                let Some(opt) = raw_opt.as_object() else { continue };
                let label = opt.get("label").and_then(Value::as_str).unwrap_or("").trim().to_owned();
                let description = opt
                    .get("description")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .trim()
                    .to_owned();
                let preview = opt
                    .get("preview")
                    .and_then(Value::as_str)
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                    .map(str::to_owned);
                if label.is_empty() {
                    continue;
                }
                options.push(AskUserQuestionOption { label, description, preview });
            }
        }
        if options.len() < 2 {
            continue;
        }
        prompts.push(AskUserQuestionPrompt { question, header, multi_select, options });
    }
    prompts
}

/// Mirrors `askUserQuestionOptions(prompt)` — synthesises wire
/// `option_id` slugs as `question_<index>` so the TUI can map back
/// to the upstream label list when responding.
#[must_use]
pub fn ask_user_question_wire_options(prompt: &AskUserQuestionPrompt) -> Vec<TuiQuestionOption> {
    prompt
        .options
        .iter()
        .enumerate()
        .map(|(i, opt)| TuiQuestionOption {
            option_id: format!("question_{i}"),
            label: opt.label.clone(),
            description: if opt.description.is_empty() { None } else { Some(opt.description.clone()) },
            preview: opt.preview.clone(),
        })
        .collect()
}

/// Mirrors `buildQuestionRequest(promptToolCall, prompt, index, total)`.
#[must_use]
pub fn build_question_request(
    base_tool_call: &ToolCall,
    prompt: &AskUserQuestionPrompt,
    index: u64,
    total: u64,
) -> QuestionRequest {
    let options = ask_user_question_wire_options(prompt);
    let prompt_tool_call = ToolCall {
        title: prompt.question.clone(),
        raw_input: Some(serde_json::json!({
            "prompt": {
                "question": prompt.question,
                "header": prompt.header,
                "multi_select": prompt.multi_select,
                "options": options,
            },
            "question_index": index,
            "total_questions": total,
        })),
        ..base_tool_call.clone()
    };
    QuestionRequest {
        tool_call: prompt_tool_call,
        prompt: QuestionPrompt {
            question: prompt.question.clone(),
            header: prompt.header.clone(),
            multi_select: prompt.multi_select,
            options,
        },
        question_index: index,
        total_questions: total,
    }
}

/// Mirrors `deriveAnnotation`. Joins option previews with a blank
/// line separator when no caller-supplied annotation preview is
/// present.
#[must_use]
pub fn derive_annotation(
    selected: &[TuiQuestionOption],
    incoming: Option<&QuestionAnnotation>,
) -> Option<QuestionAnnotation> {
    let preview_raw = incoming
        .and_then(|a| a.preview.as_deref())
        .map_or("", str::trim);
    let preview = if preview_raw.is_empty() {
        let parts: Vec<&str> = selected
            .iter()
            .filter_map(|o| o.preview.as_deref().map(str::trim).filter(|s| !s.is_empty()))
            .collect();
        if parts.is_empty() { String::new() } else { parts.join("\n\n") }
    } else {
        preview_raw.to_owned()
    };
    let notes = incoming
        .and_then(|a| a.notes.as_deref())
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_owned);
    if preview.is_empty() && notes.is_none() {
        return None;
    }
    Some(QuestionAnnotation {
        preview: if preview.is_empty() { None } else { Some(preview) },
        notes,
    })
}

/// Helper used by the worker's loop driver: build the final
/// `updatedInput` payload after every question has been answered.
/// Mirrors the tail of `requestAskUserQuestionAnswers` upstream.
#[must_use]
pub fn build_updated_input(
    original_input: &Value,
    answers: Map<String, Value>,
    annotations: Map<String, Value>,
) -> Value {
    let mut merged = original_input.as_object().cloned().unwrap_or_default();
    merged.insert("answers".to_owned(), Value::Object(answers));
    if !annotations.is_empty() {
        merged.insert("annotations".to_owned(), Value::Object(annotations));
    }
    Value::Object(merged)
}

/// Helper for the running transcript shown in the tool card while
/// each question is being asked. Mirrors `askUserQuestionTranscript`
/// upstream.
#[must_use]
pub fn ask_user_question_transcript(answers: &[(String, String, String)]) -> String {
    answers
        .iter()
        .map(|(header, question, answer)| format!("{header}: {answer}\n  {question}"))
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parse_drops_invalid_entries() {
        let input = json!({"questions": [
            { "question": "  ", "options": [{"label":"a"},{"label":"b"}] },
            { "question": "Q1", "options": [{"label":"only"}] },
            { "question": "Q2", "header": "H2", "multiSelect": true, "options": [
                {"label":"L1","description":"d1"},
                {"label":"L2","description":"","preview":"p2"},
            ]},
            "not_an_object",
        ]});
        let prompts = parse_ask_user_question_prompts(&input);
        assert_eq!(prompts.len(), 1);
        assert_eq!(prompts[0].question, "Q2");
        assert_eq!(prompts[0].header, "H2");
        assert!(prompts[0].multi_select);
        assert_eq!(prompts[0].options.len(), 2);
        assert_eq!(prompts[0].options[1].preview.as_deref(), Some("p2"));
    }

    #[test]
    fn empty_questions_yields_empty_prompts() {
        assert!(parse_ask_user_question_prompts(&json!({})).is_empty());
        assert!(parse_ask_user_question_prompts(&json!({"questions": []})).is_empty());
    }

    #[test]
    fn header_falls_back_to_q_n_when_blank() {
        let input = json!({"questions": [{
            "question": "first",
            "options": [{"label":"a"},{"label":"b"}],
        }, {
            "question": "second",
            "options": [{"label":"a"},{"label":"b"}],
        }]});
        let prompts = parse_ask_user_question_prompts(&input);
        assert_eq!(prompts[0].header, "Q1");
        assert_eq!(prompts[1].header, "Q2");
    }

    #[test]
    fn build_request_carries_index_and_total() {
        let prompt = AskUserQuestionPrompt {
            question: "Q?".to_owned(),
            header: "H".to_owned(),
            multi_select: false,
            options: vec![
                AskUserQuestionOption {
                    label: "A".to_owned(),
                    description: String::new(),
                    preview: None,
                },
                AskUserQuestionOption {
                    label: "B".to_owned(),
                    description: "desc".to_owned(),
                    preview: Some("p".to_owned()),
                },
            ],
        };
        let base = ToolCall {
            tool_call_id: "tu".to_owned(),
            title: "AskUserQuestion".to_owned(),
            kind: "ask".to_owned(),
            status: "pending".to_owned(),
            content: Vec::new(),
            raw_input: None,
            raw_output: None,
            output_metadata: None,
            task_metadata: None,
            locations: Vec::new(),
            meta: None,
        };
        let req = build_question_request(&base, &prompt, 1, 3);
        assert_eq!(req.question_index, 1);
        assert_eq!(req.total_questions, 3);
        assert_eq!(req.prompt.question, "Q?");
        assert_eq!(req.prompt.options.len(), 2);
        assert_eq!(req.prompt.options[0].option_id, "question_0");
        assert_eq!(req.prompt.options[1].option_id, "question_1");
        assert_eq!(req.prompt.options[1].preview.as_deref(), Some("p"));
        assert_eq!(req.tool_call.title, "Q?");
    }

    #[test]
    fn derive_annotation_joins_previews_when_caller_omitted() {
        let opts = vec![
            TuiQuestionOption {
                option_id: "question_0".to_owned(),
                label: "A".to_owned(),
                description: None,
                preview: Some("a-preview".to_owned()),
            },
            TuiQuestionOption {
                option_id: "question_1".to_owned(),
                label: "B".to_owned(),
                description: None,
                preview: Some("b-preview".to_owned()),
            },
        ];
        let derived = derive_annotation(&opts, None).unwrap();
        assert_eq!(derived.preview.as_deref(), Some("a-preview\n\nb-preview"));
        assert!(derived.notes.is_none());
    }

    #[test]
    fn derive_annotation_falls_through_when_nothing_to_say() {
        let opts: Vec<TuiQuestionOption> = Vec::new();
        assert!(derive_annotation(&opts, None).is_none());
    }
}
