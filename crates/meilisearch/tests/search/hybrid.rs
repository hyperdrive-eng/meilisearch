use meili_snap::snapshot;
use once_cell::sync::Lazy;

use crate::common::index::Index;
use crate::common::{GetAllDocumentsOptions, Server, Value};
use crate::json;

async fn index_with_documents_user_provided<'a>(
    server: &'a Server,
    documents: &Value,
) -> Index<'a> {
    let index = server.index("test");

    let (response, code) = index
        .update_settings(json!({ "embedders": {"default": {
                "source": "userProvided",
                "dimensions": 2}}} ))
        .await;
    assert_eq!(202, code, "{:?}", response);
    index.wait_task(response.uid()).await.succeeded();

    let (response, code) = index.add_documents(documents.clone(), None).await;
    assert_eq!(202, code, "{:?}", response);
    index.wait_task(response.uid()).await.succeeded();
    index
}

async fn index_with_documents_hf<'a>(server: &'a Server, documents: &Value) -> Index<'a> {
    let index = server.index("test");

    let (response, code) = index
        .update_settings(json!({ "embedders": {"default": {
            "source": "huggingFace",
            "model": "sentence-transformers/all-MiniLM-L6-v2",
            "revision": "e4ce9877abf3edfe10b0d82785e83bdcb973e22e",
            "documentTemplate": "{{doc.title}}, {{doc.desc}}"
        }}} ))
        .await;
    assert_eq!(202, code, "{:?}", response);
    index.wait_task(response.uid()).await.succeeded();

    let (response, code) = index.add_documents(documents.clone(), None).await;
    assert_eq!(202, code, "{:?}", response);
    index.wait_task(response.uid()).await.succeeded();
    index
}

static SIMPLE_SEARCH_DOCUMENTS_VEC: Lazy<Value> = Lazy::new(|| {
    json!([
    {
        "title": "Shazam!",
        "desc": "a Captain Marvel ersatz",
        "id": "1",
        "_vectors": {"default": [1.0, 3.0]},
    },
    {
        "title": "Captain Planet",
        "desc": "He's not part of the Marvel Cinematic Universe",
        "id": "2",
        "_vectors": {"default": [1.0, 2.0]},
    },
    {
        "title": "Captain Marvel",
        "desc": "a Shazam ersatz",
        "id": "3",
        "_vectors": {"default": [2.0, 3.0]},
    }])
});

static SINGLE_DOCUMENT_VEC: Lazy<Value> = Lazy::new(|| {
    json!([{
            "title": "Shazam!",
            "desc": "a Captain Marvel ersatz",
            "id": "1",
            "_vectors": {"default": [1.0, 3.0]},
    }])
});

static SIMPLE_SEARCH_DOCUMENTS: Lazy<Value> = Lazy::new(|| {
    json!([
    {
        "title": "Shazam!",
        "desc": "a Captain Marvel ersatz",
        "id": "1",
    },
    {
        "title": "Captain Planet",
        "desc": "He's not part of the Marvel Cinematic Universe",
        "id": "2",
    },
    {
        "title": "Captain Marvel",
        "desc": "a Shazam ersatz",
        "id": "3",
    }])
});

#[actix_rt::test]
async fn simple_search() {
    let server = Server::new().await;
    let index = index_with_documents_user_provided(&server, &SIMPLE_SEARCH_DOCUMENTS_VEC).await;

    let (response, code) = index
        .search_post(
            json!({"q": "Captain", "vector": [1.0, 1.0], "hybrid": {"semanticRatio": 0.2, "embedder": "default"}, "retrieveVectors": true}),
        )
        .await;
    snapshot!(code, @"200 OK");
    snapshot!(response["hits"], @r###"[{"title":"Captain Planet","desc":"He's not part of the Marvel Cinematic Universe","id":"2","_vectors":{"default":{"embeddings":[[1.0,2.0]],"regenerate":false}}},{"title":"Captain Marvel","desc":"a Shazam ersatz","id":"3","_vectors":{"default":{"embeddings":[[2.0,3.0]],"regenerate":false}}},{"title":"Shazam!","desc":"a Captain Marvel ersatz","id":"1","_vectors":{"default":{"embeddings":[[1.0,3.0]],"regenerate":false}}}]"###);
    snapshot!(response["semanticHitCount"], @"0");

    let (response, code) = index
        .search_post(
            json!({"q": "Captain", "vector": [1.0, 1.0], "hybrid": {"semanticRatio": 0.5, "embedder": "default"}, "showRankingScore": true, "retrieveVectors": true}),
        )
        .await;
    snapshot!(code, @"200 OK");
    snapshot!(response["hits"], @r###"[{"title":"Captain Marvel","desc":"a Shazam ersatz","id":"3","_vectors":{"default":{"embeddings":[[2.0,3.0]],"regenerate":false}},"_rankingScore":0.990290343761444},{"title":"Captain Planet","desc":"He's not part of the Marvel Cinematic Universe","id":"2","_vectors":{"default":{"embeddings":[[1.0,2.0]],"regenerate":false}},"_rankingScore":0.9848484848484848},{"title":"Shazam!","desc":"a Captain Marvel ersatz","id":"1","_vectors":{"default":{"embeddings":[[1.0,3.0]],"regenerate":false}},"_rankingScore":0.9472135901451112}]"###);
    snapshot!(response["semanticHitCount"], @"2");

    let (response, code) = index
        .search_post(
            json!({"q": "Captain", "vector": [1.0, 1.0], "hybrid": {"semanticRatio": 0.8, "embedder": "default"}, "showRankingScore": true, "retrieveVectors": true}),
        )
        .await;
    snapshot!(code, @"200 OK");
    snapshot!(response["hits"], @r###"[{"title":"Captain Marvel","desc":"a Shazam ersatz","id":"3","_vectors":{"default":{"embeddings":[[2.0,3.0]],"regenerate":false}},"_rankingScore":0.990290343761444},{"title":"Captain Planet","desc":"He's not part of the Marvel Cinematic Universe","id":"2","_vectors":{"default":{"embeddings":[[1.0,2.0]],"regenerate":false}},"_rankingScore":0.974341630935669},{"title":"Shazam!","desc":"a Captain Marvel ersatz","id":"1","_vectors":{"default":{"embeddings":[[1.0,3.0]],"regenerate":false}},"_rankingScore":0.9472135901451112}]"###);
    snapshot!(response["semanticHitCount"], @"3");
}

#[actix_rt::test]
async fn limit_offset() {
    let server = Server::new().await;
    let index = index_with_documents_user_provided(&server, &SIMPLE_SEARCH_DOCUMENTS_VEC).await;

    let (response, code) = index
        .search_post(
            json!({"q": "Captain", "vector": [1.0, 1.0], "hybrid": {"semanticRatio": 0.2, "embedder": "default"}, "retrieveVectors": true, "offset": 1, "limit": 1}),
        )
        .await;
    snapshot!(code, @"200 OK");
    snapshot!(response["hits"], @r###"[{"title":"Captain Marvel","desc":"a Shazam ersatz","id":"3","_vectors":{"default":{"embeddings":[[2.0,3.0]],"regenerate":false}}}]"###);
    snapshot!(response["semanticHitCount"], @"0");
    assert_eq!(response["hits"].as_array().unwrap().len(), 1);

    let server = Server::new().await;
    let index = index_with_documents_user_provided(&server, &SIMPLE_SEARCH_DOCUMENTS_VEC).await;

    let (response, code) = index
        .search_post(
            json!({"q": "Captain", "vector": [1.0, 1.0], "hybrid": {"semanticRatio": 0.9, "embedder": "default"}, "retrieveVectors": true, "offset": 1, "limit": 1}),
        )
        .await;
    snapshot!(code, @"200 OK");
    snapshot!(response["hits"], @r###"[{"title":"Captain Planet","desc":"He's not part of the Marvel Cinematic Universe","id":"2","_vectors":{"default":{"embeddings":[[1.0,2.0]],"regenerate":false}}}]"###);
    snapshot!(response["semanticHitCount"], @"1");
    assert_eq!(response["hits"].as_array().unwrap().len(), 1);
}

#[actix_rt::test]
async fn simple_search_hf() {
    let server = Server::new().await;
    let index = index_with_documents_hf(&server, &SIMPLE_SEARCH_DOCUMENTS).await;

    let (response, code) = index
        .search_post(
            json!({"q": "Captain", "hybrid": {"semanticRatio": 0.2, "embedder": "default"}}),
        )
        .await;
    snapshot!(code, @"200 OK");
    snapshot!(response["hits"], @r###"[{"title":"Captain Planet","desc":"He's not part of the Marvel Cinematic Universe","id":"2"},{"title":"Captain Marvel","desc":"a Shazam ersatz","id":"3"},{"title":"Shazam!","desc":"a Captain Marvel ersatz","id":"1"}]"###);
    snapshot!(response["semanticHitCount"], @"0");

    let (response, code) = index
        .search_post(
            // disable ranking score as the vectors between architectures are not equal
            json!({"q": "Captain", "hybrid": {"embedder": "default", "semanticRatio": 0.55}, "showRankingScore": false}),
        )
        .await;
    snapshot!(code, @"200 OK");
    snapshot!(response["hits"], @r###"[{"title":"Captain Planet","desc":"He's not part of the Marvel Cinematic Universe","id":"2"},{"title":"Captain Marvel","desc":"a Shazam ersatz","id":"3"},{"title":"Shazam!","desc":"a Captain Marvel ersatz","id":"1"}]"###);
    snapshot!(response["semanticHitCount"], @"1");

    let (response, code) = index
        .search_post(
            json!({"q": "Captain", "hybrid": {"embedder": "default", "semanticRatio": 0.8}, "showRankingScore": false}),
        )
        .await;
    snapshot!(code, @"200 OK");
    snapshot!(response["hits"], @r###"[{"title":"Shazam!","desc":"a Captain Marvel ersatz","id":"1"},{"title":"Captain Marvel","desc":"a Shazam ersatz","id":"3"},{"title":"Captain Planet","desc":"He's not part of the Marvel Cinematic Universe","id":"2"}]"###);
    snapshot!(response["semanticHitCount"], @"3");

    let (response, code) = index
        .search_post(
            json!({"q": "Movie World", "hybrid": {"embedder": "default", "semanticRatio": 0.2}, "showRankingScore": false}),
        )
        .await;
    snapshot!(code, @"200 OK");
    snapshot!(response["hits"], @r###"[{"title":"Captain Planet","desc":"He's not part of the Marvel Cinematic Universe","id":"2"},{"title":"Shazam!","desc":"a Captain Marvel ersatz","id":"1"},{"title":"Captain Marvel","desc":"a Shazam ersatz","id":"3"}]"###);
    snapshot!(response["semanticHitCount"], @"3");

    let (response, code) = index
        .search_post(
            json!({"q": "Wonder replacement", "hybrid": {"embedder": "default", "semanticRatio": 0.2}, "showRankingScore": false}),
        )
        .await;
    snapshot!(code, @"200 OK");
    snapshot!(response["hits"], @r###"[{"title":"Captain Marvel","desc":"a Shazam ersatz","id":"3"},{"title":"Shazam!","desc":"a Captain Marvel ersatz","id":"1"},{"title":"Captain Planet","desc":"He's not part of the Marvel Cinematic Universe","id":"2"}]"###);
    snapshot!(response["semanticHitCount"], @"3");
}

#[actix_rt::test]
async fn distribution_shift() {
    let server = Server::new().await;
    let index = index_with_documents_user_provided(&server, &SIMPLE_SEARCH_DOCUMENTS_VEC).await;

    let search = json!({"q": "Captain", "vector": [1.0, 1.0], "showRankingScore": true, "hybrid": {"embedder": "default", "semanticRatio": 1.0}, "retrieveVectors": true});
    let (response, code) = index.search_post(search.clone()).await;
    snapshot!(code, @"200 OK");
    snapshot!(response["hits"], @r###"[{"title":"Captain Marvel","desc":"a Shazam ersatz","id":"3","_vectors":{"default":{"embeddings":[[2.0,3.0]],"regenerate":false}},"_rankingScore":0.990290343761444},{"title":"Captain Planet","desc":"He's not part of the Marvel Cinematic Universe","id":"2","_vectors":{"default":{"embeddings":[[1.0,2.0]],"regenerate":false}},"_rankingScore":0.974341630935669},{"title":"Shazam!","desc":"a Captain Marvel ersatz","id":"1","_vectors":{"default":{"embeddings":[[1.0,3.0]],"regenerate":false}},"_rankingScore":0.9472135901451112}]"###);

    let (response, code) = index
        .update_settings(json!({
            "embedders": {
                "default": {
                    "distribution": {
                        "mean": 0.998,
                        "sigma": 0.01
                    }
                }
            }
        }))
        .await;

    snapshot!(code, @"202 Accepted");
    let response = server.wait_task(response.uid()).await;
    snapshot!(response["details"], @r#"{"embedders":{"default":{"distribution":{"mean":0.998,"sigma":0.01}}}}"#);

    let (response, code) = index.search_post(search).await;
    snapshot!(code, @"200 OK");
    snapshot!(response["hits"], @r###"[{"title":"Captain Marvel","desc":"a Shazam ersatz","id":"3","_vectors":{"default":{"embeddings":[[2.0,3.0]],"regenerate":false}},"_rankingScore":0.19161224365234375},{"title":"Captain Planet","desc":"He's not part of the Marvel Cinematic Universe","id":"2","_vectors":{"default":{"embeddings":[[1.0,2.0]],"regenerate":false}},"_rankingScore":1.1920928955078125e-7},{"title":"Shazam!","desc":"a Captain Marvel ersatz","id":"1","_vectors":{"default":{"embeddings":[[1.0,3.0]],"regenerate":false}},"_rankingScore":1.1920928955078125e-7}]"###);
}

#[actix_rt::test]
async fn highlighter() {
    let server = Server::new().await;
    let index = index_with_documents_user_provided(&server, &SIMPLE_SEARCH_DOCUMENTS_VEC).await;

    let (response, code) = index
        .search_post(json!({"q": "Captain Marvel", "vector": [1.0, 1.0],
            "hybrid": {"embedder": "default", "semanticRatio": 0.2},
           "retrieveVectors": true,
           "attributesToHighlight": [
                     "desc",
                     "_vectors",
                   ],
           "highlightPreTag": "**BEGIN**",
           "highlightPostTag": "**END**",
        }))
        .await;
    snapshot!(code, @"200 OK");
    snapshot!(response["hits"], @r###"[{"title":"Captain Marvel","desc":"a Shazam ersatz","id":"3","_vectors":{"default":{"embeddings":[[2.0,3.0]],"regenerate":false}},"_formatted":{"title":"Captain Marvel","desc":"a Shazam ersatz","id":"3"}},{"title":"Shazam!","desc":"a Captain Marvel ersatz","id":"1","_vectors":{"default":{"embeddings":[[1.0,3.0]],"regenerate":false}},"_formatted":{"title":"Shazam!","desc":"a **BEGIN**Captain**END** **BEGIN**Marvel**END** ersatz","id":"1"}},{"title":"Captain Planet","desc":"He's not part of the Marvel Cinematic Universe","id":"2","_vectors":{"default":{"embeddings":[[1.0,2.0]],"regenerate":false}},"_formatted":{"title":"Captain Planet","desc":"He's not part of the **BEGIN**Marvel**END** Cinematic Universe","id":"2"}}]"###);
    snapshot!(response["semanticHitCount"], @"0");

    let (response, code) = index
        .search_post(json!({"q": "Captain Marvel", "vector": [1.0, 1.0],
            "hybrid": {"embedder": "default", "semanticRatio": 0.8},
            "retrieveVectors": true,
            "showRankingScore": true,
            "attributesToHighlight": [
                     "desc"
                   ],
                   "highlightPreTag": "**BEGIN**",
                   "highlightPostTag": "**END**"
        }))
        .await;
    snapshot!(code, @"200 OK");
    snapshot!(response["hits"], @r###"[{"title":"Captain Marvel","desc":"a Shazam ersatz","id":"3","_vectors":{"default":{"embeddings":[[2.0,3.0]],"regenerate":false}},"_formatted":{"title":"Captain Marvel","desc":"a Shazam ersatz","id":"3"},"_rankingScore":0.990290343761444},{"title":"Captain Planet","desc":"He's not part of the Marvel Cinematic Universe","id":"2","_vectors":{"default":{"embeddings":[[1.0,2.0]],"regenerate":false}},"_formatted":{"title":"Captain Planet","desc":"He's not part of the **BEGIN**Marvel**END** Cinematic Universe","id":"2"},"_rankingScore":0.974341630935669},{"title":"Shazam!","desc":"a Captain Marvel ersatz","id":"1","_vectors":{"default":{"embeddings":[[1.0,3.0]],"regenerate":false}},"_formatted":{"title":"Shazam!","desc":"a **BEGIN**Captain**END** **BEGIN**Marvel**END** ersatz","id":"1"},"_rankingScore":0.9472135901451112}]"###);
    snapshot!(response["semanticHitCount"], @"3");

    // no highlighting on full semantic
    let (response, code) = index
        .search_post(json!({"q": "Captain Marvel", "vector": [1.0, 1.0],
            "hybrid": {"embedder": "default", "semanticRatio": 1.0},
            "retrieveVectors": true,
            "showRankingScore": true,
            "attributesToHighlight": [
                     "desc"
                   ],
                   "highlightPreTag": "**BEGIN**",
                   "highlightPostTag": "**END**"
        }))
        .await;
    snapshot!(code, @"200 OK");
    snapshot!(response["hits"], @r###"[{"title":"Captain Marvel","desc":"a Shazam ersatz","id":"3","_vectors":{"default":{"embeddings":[[2.0,3.0]],"regenerate":false}},"_formatted":{"title":"Captain Marvel","desc":"a Shazam ersatz","id":"3"},"_rankingScore":0.990290343761444},{"title":"Captain Planet","desc":"He's not part of the Marvel Cinematic Universe","id":"2","_vectors":{"default":{"embeddings":[[1.0,2.0]],"regenerate":false}},"_formatted":{"title":"Captain Planet","desc":"He's not part of the Marvel Cinematic Universe","id":"2"},"_rankingScore":0.974341630935669},{"title":"Shazam!","desc":"a Captain Marvel ersatz","id":"1","_vectors":{"default":{"embeddings":[[1.0,3.0]],"regenerate":false}},"_formatted":{"title":"Shazam!","desc":"a Captain Marvel ersatz","id":"1"},"_rankingScore":0.9472135901451112}]"###);
    snapshot!(response["semanticHitCount"], @"3");
}

#[actix_rt::test]
async fn invalid_semantic_ratio() {
    let server = Server::new().await;
    let index = index_with_documents_user_provided(&server, &SIMPLE_SEARCH_DOCUMENTS_VEC).await;

    let (response, code) = index
        .search_post(
            json!({"q": "Captain", "vector": [1.0, 1.0], "hybrid": {"embedder": "default", "semanticRatio": 1.2}}),
        )
        .await;
    snapshot!(code, @"400 Bad Request");
    snapshot!(response, @r###"
    {
      "message": "Invalid value at `.hybrid.semanticRatio`: the value of `semanticRatio` is invalid, expected a float between `0.0` and `1.0`.",
      "code": "invalid_search_semantic_ratio",
      "type": "invalid_request",
      "link": "https://docs.meilisearch.com/errors#invalid_search_semantic_ratio"
    }
    "###);

    let (response, code) = index
        .search_post(
            json!({"q": "Captain", "vector": [1.0, 1.0], "hybrid": {"embedder": "default", "semanticRatio": -0.8}}),
        )
        .await;
    snapshot!(code, @"400 Bad Request");
    snapshot!(response, @r###"
    {
      "message": "Invalid value at `.hybrid.semanticRatio`: the value of `semanticRatio` is invalid, expected a float between `0.0` and `1.0`.",
      "code": "invalid_search_semantic_ratio",
      "type": "invalid_request",
      "link": "https://docs.meilisearch.com/errors#invalid_search_semantic_ratio"
    }
    "###);

    let (response, code) = index
        .search_get(
            &yaup::to_string(
                &json!({"q": "Captain", "vector": [1.0, 1.0], "hybridEmbedder": "default", "hybridSemanticRatio": 1.2}),
            )
            .unwrap(),
        )
        .await;
    snapshot!(code, @"400 Bad Request");
    snapshot!(response, @r###"
    {
      "message": "Invalid value in parameter `hybridSemanticRatio`: the value of `semanticRatio` is invalid, expected a float between `0.0` and `1.0`.",
      "code": "invalid_search_semantic_ratio",
      "type": "invalid_request",
      "link": "https://docs.meilisearch.com/errors#invalid_search_semantic_ratio"
    }
    "###);

    let (response, code) = index
        .search_get(
            &yaup::to_string(
                &json!({"q": "Captain", "vector": [1.0, 1.0], "hybridEmbedder": "default", "hybridSemanticRatio": -0.2}),
            )
            .unwrap(),
        )
        .await;
    snapshot!(code, @"400 Bad Request");
    snapshot!(response, @r###"
    {
      "message": "Invalid value in parameter `hybridSemanticRatio`: the value of `semanticRatio` is invalid, expected a float between `0.0` and `1.0`.",
      "code": "invalid_search_semantic_ratio",
      "type": "invalid_request",
      "link": "https://docs.meilisearch.com/errors#invalid_search_semantic_ratio"
    }
    "###);
}

#[actix_rt::test]
async fn single_document() {
    let server = Server::new().await;
    let index = index_with_documents_user_provided(&server, &SINGLE_DOCUMENT_VEC).await;

    let (response, code) = index
    .search_post(
        json!({"vector": [1.0, 3.0], "hybrid": {"semanticRatio": 1.0, "embedder": "default"}, "showRankingScore": true, "retrieveVectors": true}),
    )
    .await;

    snapshot!(code, @"200 OK");
    snapshot!(response["hits"][0], @r###"{"title":"Shazam!","desc":"a Captain Marvel ersatz","id":"1","_vectors":{"default":{"embeddings":[[1.0,3.0]],"regenerate":false}},"_rankingScore":1.0}"###);
    snapshot!(response["semanticHitCount"], @"1");
}

#[actix_rt::test]
async fn query_combination() {
    let server = Server::new().await;
    let index = index_with_documents_user_provided(&server, &SIMPLE_SEARCH_DOCUMENTS_VEC).await;

    // search without query and vector, but with hybrid => still placeholder
    let (response, code) = index
        .search_post(json!({"hybrid": {"embedder": "default", "semanticRatio": 1.0}, "showRankingScore": true, "retrieveVectors": true}))
        .await;

    snapshot!(code, @"200 OK");
    snapshot!(response["hits"], @r###"[{"title":"Shazam!","desc":"a Captain Marvel ersatz","id":"1","_vectors":{"default":{"embeddings":[[1.0,3.0]],"regenerate":false}},"_rankingScore":1.0},{"title":"Captain Planet","desc":"He's not part of the Marvel Cinematic Universe","id":"2","_vectors":{"default":{"embeddings":[[1.0,2.0]],"regenerate":false}},"_rankingScore":1.0},{"title":"Captain Marvel","desc":"a Shazam ersatz","id":"3","_vectors":{"default":{"embeddings":[[2.0,3.0]],"regenerate":false}},"_rankingScore":1.0}]"###);
    snapshot!(response["semanticHitCount"], @"null");

    // same with a different semantic ratio
    let (response, code) = index
        .search_post(json!({"hybrid": {"embedder": "default", "semanticRatio": 0.76}, "showRankingScore": true, "retrieveVectors": true}))
        .await;

    snapshot!(code, @"200 OK");
    snapshot!(response["hits"], @r###"[{"title":"Shazam!","desc":"a Captain Marvel ersatz","id":"1","_vectors":{"default":{"embeddings":[[1.0,3.0]],"regenerate":false}},"_rankingScore":1.0},{"title":"Captain Planet","desc":"He's not part of the Marvel Cinematic Universe","id":"2","_vectors":{"default":{"embeddings":[[1.0,2.0]],"regenerate":false}},"_rankingScore":1.0},{"title":"Captain Marvel","desc":"a Shazam ersatz","id":"3","_vectors":{"default":{"embeddings":[[2.0,3.0]],"regenerate":false}},"_rankingScore":1.0}]"###);
    snapshot!(response["semanticHitCount"], @"null");

    // wrong vector dimensions
    let (response, code) = index
    .search_post(json!({"vector": [1.0, 0.0, 1.0], "hybrid": {"embedder": "default", "semanticRatio": 1.0}, "showRankingScore": true, "retrieveVectors": true}))
    .await;

    snapshot!(code, @"400 Bad Request");
    snapshot!(response, @r###"
    {
      "message": "Invalid vector dimensions: expected: `2`, found: `3`.",
      "code": "invalid_vector_dimensions",
      "type": "invalid_request",
      "link": "https://docs.meilisearch.com/errors#invalid_vector_dimensions"
    }
    "###);

    // full vector
    let (response, code) = index
    .search_post(json!({"vector": [1.0, 0.0], "hybrid": {"embedder": "default", "semanticRatio": 1.0}, "showRankingScore": true, "retrieveVectors": true}))
    .await;

    snapshot!(code, @"200 OK");
    snapshot!(response["hits"], @r###"[{"title":"Captain Marvel","desc":"a Shazam ersatz","id":"3","_vectors":{"default":{"embeddings":[[2.0,3.0]],"regenerate":false}},"_rankingScore":0.7773500680923462},{"title":"Captain Planet","desc":"He's not part of the Marvel Cinematic Universe","id":"2","_vectors":{"default":{"embeddings":[[1.0,2.0]],"regenerate":false}},"_rankingScore":0.7236068248748779},{"title":"Shazam!","desc":"a Captain Marvel ersatz","id":"1","_vectors":{"default":{"embeddings":[[1.0,3.0]],"regenerate":false}},"_rankingScore":0.6581138968467712}]"###);
    snapshot!(response["semanticHitCount"], @"3");

    // full keyword, without a query
    let (response, code) = index
    .search_post(json!({"vector": [1.0, 0.0], "hybrid": {"embedder": "default", "semanticRatio": 0.0}, "showRankingScore": true, "retrieveVectors": true}))
    .await;

    snapshot!(code, @"200 OK");
    snapshot!(response["hits"], @r###"[{"title":"Shazam!","desc":"a Captain Marvel ersatz","id":"1","_vectors":{"default":{"embeddings":[[1.0,3.0]],"regenerate":false}},"_rankingScore":1.0},{"title":"Captain Planet","desc":"He's not part of the Marvel Cinematic Universe","id":"2","_vectors":{"default":{"embeddings":[[1.0,2.0]],"regenerate":false}},"_rankingScore":1.0},{"title":"Captain Marvel","desc":"a Shazam ersatz","id":"3","_vectors":{"default":{"embeddings":[[2.0,3.0]],"regenerate":false}},"_rankingScore":1.0}]"###);
    snapshot!(response["semanticHitCount"], @"null");

    // query + vector, full keyword => keyword
    let (response, code) = index
    .search_post(json!({"q": "Captain", "vector": [1.0, 0.0], "hybrid": {"embedder": "default", "semanticRatio": 0.0}, "showRankingScore": true, "retrieveVectors": true}))
    .await;

    snapshot!(code, @"200 OK");
    snapshot!(response["hits"], @r###"[{"title":"Captain Planet","desc":"He's not part of the Marvel Cinematic Universe","id":"2","_vectors":{"default":{"embeddings":[[1.0,2.0]],"regenerate":false}},"_rankingScore":0.9848484848484848},{"title":"Captain Marvel","desc":"a Shazam ersatz","id":"3","_vectors":{"default":{"embeddings":[[2.0,3.0]],"regenerate":false}},"_rankingScore":0.9848484848484848},{"title":"Shazam!","desc":"a Captain Marvel ersatz","id":"1","_vectors":{"default":{"embeddings":[[1.0,3.0]],"regenerate":false}},"_rankingScore":0.9242424242424242}]"###);
    snapshot!(response["semanticHitCount"], @"null");

    // query + vector, no hybrid keyword =>
    let (response, code) = index
        .search_post(json!({"q": "Captain", "vector": [1.0, 0.0], "showRankingScore": true, "retrieveVectors": true}))
        .await;

    snapshot!(code, @"400 Bad Request");
    snapshot!(response, @r###"
    {
      "message": "Invalid request: missing `hybrid` parameter when `vector` is present.",
      "code": "missing_search_hybrid",
      "type": "invalid_request",
      "link": "https://docs.meilisearch.com/errors#missing_search_hybrid"
    }
    "###);

    // full vector, without a vector => error
    let (response, code) = index
        .search_post(
            json!({"q": "Captain", "hybrid": {"semanticRatio": 1.0, "embedder": "default"}, "showRankingScore": true, "retrieveVectors": true}),
        )
        .await;

    snapshot!(code, @"400 Bad Request");
    snapshot!(response, @r###"
    {
      "message": "Error while generating embeddings: user error: attempt to embed the following text in a configuration where embeddings must be user provided:\n  - `Captain`",
      "code": "vector_embedding_error",
      "type": "invalid_request",
      "link": "https://docs.meilisearch.com/errors#vector_embedding_error"
    }
    "###);

    // hybrid without a vector => full keyword
    let (response, code) = index
        .search_post(
            json!({"q": "Planet", "hybrid": {"semanticRatio": 0.99, "embedder": "default"}, "showRankingScore": true, "retrieveVectors": true}),
        )
        .await;

    snapshot!(code, @"200 OK");
    snapshot!(response["hits"], @r###"[{"title":"Captain Planet","desc":"He's not part of the Marvel Cinematic Universe","id":"2","_vectors":{"default":{"embeddings":[[1.0,2.0]],"regenerate":false}},"_rankingScore":0.9242424242424242}]"###);
    snapshot!(response["semanticHitCount"], @"0");
}

#[actix_rt::test]
async fn retrieve_vectors() {
    let server = Server::new().await;
    let index = index_with_documents_hf(&server, &SIMPLE_SEARCH_DOCUMENTS).await;

    let (response, code) = index
        .search_post(
            json!({"q": "Captain", "hybrid": {"embedder": "default", "semanticRatio": 0.2}, "retrieveVectors": true}),
        )
        .await;
    snapshot!(code, @"200 OK");
    insta::assert_json_snapshot!(response["hits"], {"[]._vectors.default.embeddings" => "[vectors]"},  @r###"
    [
      {
        "title": "Captain Planet",
        "desc": "He's not part of the Marvel Cinematic Universe",
        "id": "2",
        "_vectors": {
          "default": {
            "embeddings": "[vectors]",
            "regenerate": true
          }
        }
      },
      {
        "title": "Captain Marvel",
        "desc": "a Shazam ersatz",
        "id": "3",
        "_vectors": {
          "default": {
            "embeddings": "[vectors]",
            "regenerate": true
          }
        }
      },
      {
        "title": "Shazam!",
        "desc": "a Captain Marvel ersatz",
        "id": "1",
        "_vectors": {
          "default": {
            "embeddings": "[vectors]",
            "regenerate": true
          }
        }
      }
    ]
    "###);

    // use explicit `_vectors` in displayed attributes
    let (response, code) = index
        .update_settings(json!({ "displayedAttributes": ["id", "title", "desc", "_vectors"]} ))
        .await;
    assert_eq!(202, code, "{:?}", response);
    index.wait_task(response.uid()).await.succeeded();

    let (response, code) = index
        .search_post(
            json!({"q": "Captain", "hybrid": {"embedder": "default", "semanticRatio": 0.2}, "retrieveVectors": true}),
        )
        .await;
    snapshot!(code, @"200 OK");
    insta::assert_json_snapshot!(response["hits"], {"[]._vectors.default.embeddings" => "[vectors]"},  @r###"
    [
      {
        "title": "Captain Planet",
        "desc": "He's not part of the Marvel Cinematic Universe",
        "id": "2",
        "_vectors": {
          "default": {
            "embeddings": "[vectors]",
            "regenerate": true
          }
        }
      },
      {
        "title": "Captain Marvel",
        "desc": "a Shazam ersatz",
        "id": "3",
        "_vectors": {
          "default": {
            "embeddings": "[vectors]",
            "regenerate": true
          }
        }
      },
      {
        "title": "Shazam!",
        "desc": "a Captain Marvel ersatz",
        "id": "1",
        "_vectors": {
          "default": {
            "embeddings": "[vectors]",
            "regenerate": true
          }
        }
      }
    ]
    "###);

    // remove `_vectors` from displayed attributes
    let (response, code) =
        index.update_settings(json!({ "displayedAttributes": ["id", "title", "desc"]} )).await;
    assert_eq!(202, code, "{:?}", response);
    index.wait_task(response.uid()).await.succeeded();

    let (response, code) = index
        .search_post(
            json!({"q": "Captain", "hybrid": {"embedder": "default", "semanticRatio": 0.2}, "retrieveVectors": true}),
        )
        .await;
    snapshot!(code, @"200 OK");
    insta::assert_json_snapshot!(response["hits"], {"[]._vectors.default.embeddings" => "[vectors]"},  @r###"
    [
      {
        "title": "Captain Planet",
        "desc": "He's not part of the Marvel Cinematic Universe",
        "id": "2"
      },
      {
        "title": "Captain Marvel",
        "desc": "a Shazam ersatz",
        "id": "3"
      },
      {
        "title": "Shazam!",
        "desc": "a Captain Marvel ersatz",
        "id": "1"
      }
    ]
    "###);
}

// Let's modify the test to be more explicit about reproducing the bug
#[actix_rt::test]
async fn test_hybrid_search_distinct_bug_reproduction() {
    use std::collections::HashSet;
    
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

    // Create documents designed to trigger the bug:
    // - Same product_id but very different content that will rank differently
    // - Ensure one ranks high in keyword search, other in vector search
    let test_documents = json!([
        {
            "id": 1,
            "title": "NIKE SPORTS JACKET LEATHER RED",  // High keyword match
            "brand": "Nike", 
            "product_id": "SAME_PRODUCT",
            "category": "sports",
            "_vectors": {"default": [0.1, 0.1]}  // Low vector similarity
        },
        {
            "id": 2,
            "title": "Blue casual wear item", // Low keyword match
            "brand": "Nike",
            "product_id": "SAME_PRODUCT", // SAME product_id!
            "category": "fashion",
            "_vectors": {"default": [0.9, 0.9]} // High vector similarity to search vector
        },
        {
            "id": 3,
            "title": "Different product entirely",
            "brand": "Adidas",
            "product_id": "DIFFERENT_PRODUCT",
            "category": "other",
            "_vectors": {"default": [0.5, 0.5]}
        }
    ]);

    let (response, code) = index.add_documents(test_documents, Some("id")).await;
    assert_eq!(202, code);
    index.wait_task(response.uid()).await.succeeded();

    // Set distinct on product_id
    let (task, _) = index.update_distinct_attribute(json!("product_id")).await;
    index.wait_task(task.uid()).await.succeeded();

    // Search that should favor document 1 for keywords and document 2 for vectors
    let (response, code) = index
        .search_post(json!({
            "q": "NIKE SPORTS JACKET LEATHER RED",  // Should strongly match doc 1
            "vector": [0.9, 0.9],  // Should strongly match doc 2
            "hybrid": {
                "embedder": "default", 
                "semanticRatio": 0.5  // Equal weight to both searches
            },
            "limit": 10
        }))
        .await;

    assert_eq!(200, code);
    let hits = response["hits"].as_array().unwrap();
    
    println!("Total hits: {}", hits.len());
    for (i, hit) in hits.iter().enumerate() {
        println!("Hit {}: id={}, product_id={}, title={}", 
                i, 
                hit["id"], 
                hit["product_id"], 
                hit["title"]);
    }
    
    // Count occurrences of each product_id
    let mut product_id_counts = std::collections::HashMap::new();
    for hit in hits {
        if let Some(product_id) = hit["product_id"].as_str() {
            *product_id_counts.entry(product_id).or_insert(0) += 1;
        }
    }
    
    println!("Product ID counts: {:?}", product_id_counts);
    
    // Check if SAME_PRODUCT appears more than once - this would be the bug
    if let Some(&count) = product_id_counts.get("SAME_PRODUCT") {
        if count > 1 {
            panic!("BUG REPRODUCED! product_id 'SAME_PRODUCT' appears {} times in hybrid search results, violating distinct constraint", count);
        }
    }
    
    println!("Distinct filtering appears to be working correctly");
}

#[actix_rt::test]
async fn test_hybrid_distinct_bug_extreme() {
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

    // Documents with IDENTICAL distinct values that should appear in BOTH searches
    let test_documents = json!([
        {
            "id": 1,
            "title": "red shoes nike",  // Should match keyword search well
            "product_id": "DUPLICATE_PRODUCT_ID",
            "_vectors": {"default": [0.0, 0.0]}  // Low similarity to search vector
        },
        {
            "id": 2,
            "title": "green shirt adidas",  // Different content
            "product_id": "DUPLICATE_PRODUCT_ID",  // SAME product_id!
            "_vectors": {"default": [1.0, 1.0]}   // High similarity to search vector
        }
    ]);

    let (response, code) = index.add_documents(test_documents, Some("id")).await;
    assert_eq!(202, code);
    index.wait_task(response.uid()).await.succeeded();

    // Set distinct on product_id
    let (task, _) = index.update_distinct_attribute(json!("product_id")).await;
    index.wait_task(task.uid()).await.succeeded();

    // Hybrid search - should trigger both vector and keyword search
    let (response, code) = index
        .search_post(json!({
            "q": "red shoes nike",      // Should rank doc 1 high in keyword search
            "vector": [1.0, 1.0],       // Should rank doc 2 high in vector search  
            "hybrid": {
                "embedder": "default", 
                "semanticRatio": 0.5
            },
            "limit": 10
        }))
        .await;

    assert_eq!(200, code);
    let hits = response["hits"].as_array().unwrap();
    
    println!("SEARCH RESULTS:");
    for (i, hit) in hits.iter().enumerate() {
        println!("  Hit {}: id={}, product_id={}", i, hit["id"], hit["product_id"]);
    }
    
    // Check for duplicate product_ids - THIS IS THE BUG
    let product_ids: Vec<&str> = hits.iter()
        .filter_map(|hit| hit["product_id"].as_str())
        .collect();
    
    let unique_count = product_ids.iter().collect::<std::collections::HashSet<_>>().len();
    
    if product_ids.len() > unique_count {
        panic!("BUG REPRODUCED! Found {} total results but only {} unique product_ids. \
               This means distinct filtering failed in hybrid search! Results: {:?}", 
               product_ids.len(), unique_count, product_ids);
    }
    
    // If only one result, distinct filtering worked (but we hoped to reproduce the bug)
    if hits.len() == 1 {
        println!("Only one result returned - distinct filtering working or different issue");
    } else {
        println!("Multiple results with unique product_ids - distinct filtering appears to work");
    }
}

#[actix_rt::test]
async fn test_distinct_filtering_with_federated_search() {
    let server = Server::new().await;
    
    // Create two indexes
    let products_index = server.index("products");
    let categories_index = server.index("categories");

    // Set up embedder for both indexes
    for index in [&products_index, &categories_index] {
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
    }

    // Add documents to products index
    let products_documents = json!([
        {
            "id": 1,
            "title": "Red Nike Shoes",
            "product_id": "ABC123",
            "_vectors": {"default": [0.1, 0.1]}
        }
    ]);

    let (response, code) = products_index.add_documents(products_documents, Some("id")).await;
    assert_eq!(202, code);
    products_index.wait_task(response.uid()).await.succeeded();

    // Add documents to categories index
    let categories_documents = json!([
        {
            "id": 1,
            "name": "Nike Shoes",
            "category_id": "ABC123",  // Same ID as the product!
            "_vectors": {"default": [0.9, 0.9]}
        }
    ]);

    let (response, code) = categories_index.add_documents(categories_documents, Some("id")).await;
    assert_eq!(202, code);
    categories_index.wait_task(response.uid()).await.succeeded();

    // Set distinct attributes for both indexes
    for index in [&products_index, &categories_index] {
        let (task, _) = index.update_distinct_attribute(json!("product_id")).await;
        index.wait_task(task.uid()).await.succeeded();
    }

    // Perform federated search
    let (response, code) = server
        .multi_search(json!({
            "federation": {},  // Enable federation
            "queries": [
                {
                    "indexUid": "products",
                    "q": "red shoes",
                    "vector": [0.9, 0.9],
                    "hybrid": {
                        "embedder": "default",
                        "semanticRatio": 0.5
                    }
                },
                {
                    "indexUid": "categories",
                    "q": "red shoes",
                    "vector": [0.9, 0.9],
                    "hybrid": {
                        "embedder": "default",
                        "semanticRatio": 0.5
                    }
                }
            ]
        }))
        .await;
    assert_eq!(200, code);
    
    // Check results from federated search
    let hits = response["hits"].as_array().unwrap();
    
    println!("\nFEDERATED SEARCH RESULTS:");
    for (i, hit) in hits.iter().enumerate() {
        println!("  {}. id={}, index={}, federation={}", 
                i+1, 
                hit["id"], 
                hit["_federation"]["indexUid"],
                hit["_federation"]);
    }
    
    // Collect all distinct values
    let mut all_distinct_values = Vec::new();
    
    // Add product_ids from products index
    for hit in hits {
        if let Some(product_id) = hit["product_id"].as_str() {
            all_distinct_values.push(product_id);
        }
        if let Some(category_id) = hit["category_id"].as_str() {
            all_distinct_values.push(category_id);
        }
    }
    
    let unique_distinct_values: std::collections::HashSet<_> = all_distinct_values.iter().collect();
    
    println!("\nTotal results: {}, Unique distinct values: {}", 
             all_distinct_values.len(), 
             unique_distinct_values.len());
    println!("Distinct values found: {:?}", all_distinct_values);
    
    // The bug would manifest as having more results than unique distinct values
    if all_distinct_values.len() > unique_distinct_values.len() {
        panic!("BUG REPRODUCED! Federated search returned {} documents but only {} unique distinct values. \
               This proves distinct filtering is not applied after merging results from different indexes. \
               Distinct values: {:?}", 
               all_distinct_values.len(), 
               unique_distinct_values.len(), 
               all_distinct_values);
    }
    
    println!("\nTest completed. Expected 1 unique distinct value, got {}", unique_distinct_values.len());
    if unique_distinct_values.len() == 1 {
        println!("Distinct filtering appears to be working correctly in this version");
    }
}
