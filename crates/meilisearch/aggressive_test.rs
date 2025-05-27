#[actix_rt::test]
async fn test_distinct_bug_with_semantic_ratio_extremes() {
    let server = Server::new().await;
    let index = server.index("test");

    // Set up embedder
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
    assert_eq!(202, code);
    index.wait_task(response.uid()).await.succeeded();

    // Add many documents with same product_id to increase chance of bug
    let test_documents = json!([
        {
            "id": 1,
            "title": "amazing fantastic super best quality nike shoes",
            "product_id": "SAME_ID",
            "_vectors": {"default": [0.0, 0.0]}
        },
        {
            "id": 2,
            "title": "terrible awful worst product ever",
            "product_id": "SAME_ID",
            "_vectors": {"default": [1.0, 1.0]}
        },
        {
            "id": 3,
            "title": "medium average okay product",
            "product_id": "SAME_ID",
            "_vectors": {"default": [0.5, 0.5]}
        },
        {
            "id": 4,
            "title": "excellent superb outstanding item",
            "product_id": "SAME_ID",
            "_vectors": {"default": [0.2, 0.8]}
        },
        {
            "id": 5,
            "title": "poor quality bad item",
            "product_id": "SAME_ID",
            "_vectors": {"default": [0.8, 0.2]}
        }
    ]);

    let (response, code) = index.add_documents(test_documents, Some("id")).await;
    assert_eq!(202, code);
    index.wait_task(response.uid()).await.succeeded();

    // Set distinct on product_id
    let (task, _) = index.update_distinct_attribute(json!("product_id")).await;
    index.wait_task(task.uid()).await.succeeded();

    // Search with very low semantic ratio to force keyword prominence
    let (response, code) = index
        .search_post(json!({
            "q": "amazing fantastic super best quality nike shoes",
            "vector": [1.0, 1.0],
            "hybrid": {
                "embedder": "default", 
                "semanticRatio": 0.1  // Very low - keyword search dominates
            },
            "limit": 10
        }))
        .await;

    assert_eq!(200, code);
    let hits = response["hits"].as_array().unwrap();
    
    println!("LOW SEMANTIC RATIO RESULTS:");
    for hit in hits {
        println!("  id={}, product_id={}, title={}", hit["id"], hit["product_id"], hit["title"]);
    }
    
    // Search with very high semantic ratio to force vector prominence
    let (response, code) = index
        .search_post(json!({
            "q": "amazing fantastic super best quality nike shoes",
            "vector": [1.0, 1.0],
            "hybrid": {
                "embedder": "default", 
                "semanticRatio": 0.9  // Very high - vector search dominates
            },
            "limit": 10
        }))
        .await;

    assert_eq!(200, code);
    let hits = response["hits"].as_array().unwrap();
    
    println!("HIGH SEMANTIC RATIO RESULTS:");
    for hit in hits {
        println!("  id={}, product_id={}, title={}", hit["id"], hit["product_id"], hit["title"]);
    }
    
    // Now try with equal weighting
    let (response, code) = index
        .search_post(json!({
            "q": "amazing fantastic super best quality nike shoes",
            "vector": [1.0, 1.0],
            "hybrid": {
                "embedder": "default", 
                "semanticRatio": 0.5
            },
            "limit": 10
        }))
        .await;

    assert_eq!(200, code);
    let hits = response["hits"].as_array().unwrap();
    
    println!("EQUAL WEIGHTING RESULTS (should show distinct bug if it exists):");
    let product_ids: Vec<&str> = hits.iter()
        .filter_map(|hit| hit["product_id"].as_str())
        .collect();
    
    for hit in hits {
        println!("  id={}, product_id={}, title={}", hit["id"], hit["product_id"], hit["title"]);
    }
    
    let unique_product_ids: std::collections::HashSet<_> = product_ids.iter().collect();
    
    if product_ids.len() > unique_product_ids.len() {
        panic!("BUG REPRODUCED! Found {} total results but only {} unique product_ids: {:?}", 
               product_ids.len(), unique_product_ids.len(), product_ids);
    }
    
    if hits.len() > 1 {
        println!("Multiple results but all unique product_ids - distinct working");
    } else {
        println!("Only one result - either distinct working or needs more aggressive test");
    }
}
