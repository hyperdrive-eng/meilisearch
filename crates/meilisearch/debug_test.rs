use std::process::Command;

#[test] 
fn debug_hybrid_search() {
    // Let's create a debug test to understand what's happening
    let output = Command::new("curl")
        .arg("-X")
        .arg("POST")
        .arg("http://localhost:7700/indexes/test/search")
        .arg("-H")
        .arg("Content-Type: application/json")
        .arg("-d")
        .arg(r#"{"q": "leather jacket Nike", "vector": [0.5, 0.5], "hybrid": {"embedder": "default", "semanticRatio": 0.5}, "limit": 10}"#)
        .output()
        .expect("Failed to execute curl");
        
    println!("Debug output: {:?}", String::from_utf8_lossy(&output.stdout));
}
