from qdrant_client import QdrantClient

client = QdrantClient(
    url="http://qdrant-mad.hubabpe0f6d5ccd8.centralus.azurecontainer.io:6333/"
)
# client.create_collection(collection_name="test_collection")
collections = client.get_collections()
print(collections)

# IF Eval LeaderBoard
