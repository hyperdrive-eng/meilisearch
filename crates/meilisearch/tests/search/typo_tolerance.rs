use meili_snap::{json_string, snapshot};
use crate::common::Server;
use crate::json;

#[actix_rt::test]
async fn test_typo_tolerance_with_wildcard_searchable_attributes() {
    let server = Server::new().await;
    let index = server.index("test");

    // Add documents with a field that will have typo tolerance disabled
    let documents = json!([
        {
            "id": 1,
            "title": "Captain Marvel",
            "description": "A superhero movie",
            "exact_field": "Capitan Marivel" // Intentional typo
        },
        {
            "id": 2,
            "title": "Captain Marivel",
            "description": "Another movie with a typo",
            "exact_field": "Captain Marvel"
        }
    ]);
    
    index.add_documents(documents, None).await;
    index.wait_task(0).await;

    // Configure typo tolerance to disable typos on exact_field
    let (response, code) = index.update_settings_typo_tolerance(json!({
        "disableOnAttributes": ["exact_field"]
    })).await;
    assert_eq!(202, code, "{:?}", response);
    index.wait_task(1).await;

    // Test 1: With wildcard searchable attributes - BUG: Should only find document #2 without typo,
    // but finds both documents due to the bug.
    index.search(json!({
        "q": "Marvel",
        "attributesToSearchOn": ["*"]
    }), |response, code| {
        assert_eq!(200, code.as_u16());
        
        // Due to the bug, this will find both documents (id 1 and 2) because the typo tolerance
        // is not properly disabled on exact_field when using wildcard
        assert_eq!(response["hits"].as_array().unwrap().len(), 2);
    }).await;

    // Test 2: With explicit searchable attributes - Works correctly
    index.search(json!({
        "q": "Marvel",
        "attributesToSearchOn": ["title", "description", "exact_field"]
    }), |response, code| {
        assert_eq!(200, code.as_u16());
        
        // When explicitly listing the attributes, typo tolerance is correctly disabled
        // and only document #2 is found
        assert_eq!(response["hits"].as_array().unwrap().len(), 1);
        assert_eq!(response["hits"][0]["id"], 2);
    }).await;
} 