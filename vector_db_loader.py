from qdrant_client import QdrantClient

client = QdrantClient(url = "http://qdrant-mad.hubabpe0f6d5ccd8.centralus.azurecontainer.io:6333/")
# client.create_collection(collection_name="test_collection")
collections = client.get_collections()
print(collections)



# docker pull public.ecr.aws/q9t5s3a7/vllm-cpu-release-repo:v0.10.2