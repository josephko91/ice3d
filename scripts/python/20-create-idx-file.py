from sklearn.model_selection import train_test_split


indices = list(range(7_000_000))
train_idx, temp_idx = train_test_split(indices, test_size=0.30, random_state=random_state)
val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, 
random_state=random_state)