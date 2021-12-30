import pickle

for i in range(40):
    with open(f"./data/games/1_game_800_gen_0_thread_{i}.pkl", "rb") as f:
        data = pickle.load(f)
        all_data = []
        for d in data:
            all_data.extend(d)
        print(len(data))
        print(len(all_data))
    with open(f"./data/games/1_game_800_gen_0_thread_{i}.pkl", "wb") as f:
        pickle.dump(all_data, f)