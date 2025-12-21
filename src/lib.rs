use serde::{Deserialize, Serialize};

pub mod plan;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct AssignmentRequest {
    pub dataset: String,
    pub objects: Vec<String>,
    pub chunks: Vec<String>,
}
