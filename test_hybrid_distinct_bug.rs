use meili_snap::snapshot;
use once_cell::sync::Lazy;

use crate::common::{Server, Value};
use crate::json;

// Documents with same distinct value (product_id) but different characteristics
// to ensure they appear in different search types (vector vs keyword)
static TEST_DOCUMENTS: Lazy<Value> = Lazy::new(|| {
    json!([
        {
            "id": 1,
            "title": "Red leather jacket from Nike",
            "brand": "Nike", 
            "product_id": "12345",
            "description": "sportswear athletic gear",
            "_vectors": {"default": [1.0, 0.0]}
        },
        {
            "id": 2,
            "title": "Blue leather jacket from Nike", 
            "brand": "Nike",
            "product_id": "12345", // Same product_id as document 1
            "description": "fashion casual wear",
            "_vectors": {"default": [0.0, 1.0]} // Different vector to ensure different ranking
        },
        {
            "id": 3,
            "title": "Green cotton shirt",
            "brand": "Adidas",
            "product_id": "67890",
            "description": "comfortable cotton material",
            "_vectors": {"default": [1.0, 1.0]}
        }
    ])
});

#[actix_rt::test]
async fn test_hybrid_search_with_distinct_filtering_bug() {
    let server = Server::new().await;
    let index = server.index("test");

    // Set up embedder for hybrid search
    let (response, code) = index
        .update_settings(json!({ 
            "embedders": {
                "default": {
                    "source": "userProvided",
                    "dimensions": 2
                }
            }
        }))
        .await;
    assert_eq!(202, code, "{:?}", response);
    index.wait_task(response.uid()).await.succeeded();

    // Add documents
    let (response, code) = index.add_documents(TEST_DOCUMENTS.clone(), Some("id")).await;
    assert_eq!(202, code, "{:?}", response);
    index.wait_task(response.uid()).await.succeeded();

    // Set distinct attribute to product_id
    let (task, _status_code) = index.update_distinct_attribute(json!("product_id")).await;
    index.wait_task(task.uid()).await.succeeded();

    // Perform hybrid search that should return only one document per product_id
    // This search should trigger both vector and keyword search, potentially
    // returning documents with same product_id from different search types
    let (response, code) = index
        .search_post(json!({
            "q": "leather jacket Nike",
            "vector": [0.5, 0.5],
            "hybrid": {
                "embedder": "default", 
                "semanticRatio": 0.5
            },
            "limit": 10
        }))
        .await;

    assert_eq!(200, code, "{:?}", response);
    
    let hits = response["hits"].as_array().unwrap();
    
    // Extract product_ids from results
    let mut product_ids = Vec::new();
    for hit in hits {
        if let Some(product_id) = hit["product_id"].as_str() {
            product_ids.push(product_id);
        }
    }
    
    // Check for duplicates - this should fail due to the bug
    let mut unique_product_ids = std::collections::HashSet::new();
    for product_id in &product_ids {
        if !unique_product_ids.insert(product_id) {
            panic!(
                "Distinct filtering failed: product_id '{}' appears multiple times in hybrid search results. \
                Results: {:?}. This indicates the bug where distinct filtering is not applied when merging \
                results from vector search and keyword search.", 
                product_id, product_ids
            );
        }
    }

    // If we reach here without panicking, the bug is fixed
    println!("Distinct filtering working correctly in hybrid search");
}
