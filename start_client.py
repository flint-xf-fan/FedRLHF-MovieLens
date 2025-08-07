import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import flwr as fl
from fed_rlhf.client import MovieLensClient
from data.movielens import MovieLensDataset
import torch

def main():
    user_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    dataset = MovieLensDataset(ratings_path='datasets/ml-latest-small/ratings.csv',
                               movies_path='datasets/ml-latest-small/movies.csv')
    client = MovieLensClient(user_id=user_id, dataset=dataset, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Client created with user_id: {user_id}")
    # fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)
    fl.client.start_client(server_address="127.0.0.1:8081", client=client.to_client())


if __name__ == "__main__":
    main()


