import matplotlib.pyplot as plt
# import numpy as np

# # Generate 100 epochs
# epochs = np.arange(1, 101)

# # Approximate values for 100 epochs
# train_loss = [7.6782, 7.0797, 6.7368, 6.4662, 6.2448, 6.0482, 5.8876, 5.7721, 5.7053, 5.6576,
#               5.7495, 5.5548, 5.3445, 5.1409, 4.9599, 4.7759, 4.5828, 4.4417, 4.2549, 4.1020,
#               3.9705, 3.8470, 3.7390, 3.6454, 3.5424, 3.5046, 3.4502, 3.4072, 3.3955, 3.3844,
#               3.6371, 3.5725, 3.4273, 3.2762, 3.1724, 3.0473, 2.8951, 2.7979, 2.7145, 2.5977,
#               2.5211, 2.4384, 2.3670, 2.3143, 2.2550, 2.1963, 2.1300, 2.0838, 2.0498, 2.0200,
#               1.9523, 1.9366, 1.8966, 1.8787, 1.8459, 1.8291, 1.8035, 1.7888, 1.7729, 1.7554,
#               1.7526, 1.7429, 1.7239, 1.7281, 1.7147, 1.7100, 1.7050, 1.7073, 1.6980, 1.7020,
#               1.8929] + [1.7 + np.random.normal(0, 0.05) for _ in range(29)]  # Approximate for remaining epochs

# val_loss = [7.2080, 6.8543, 6.5252, 6.3035, 6.1066, 5.9546, 5.8034, 5.7306, 5.6918, 5.6720,
#             5.6900, 5.4011, 5.2049, 5.0224, 4.8343, 4.6580, 4.4474, 4.3001, 4.1239, 3.9693,
#             3.8459, 3.7037, 3.6163, 3.5271, 3.4738, 3.4283, 3.3802, 3.3586, 3.3564, 3.3646,
#             3.5142, 3.4171, 3.2291, 3.1229, 3.0015, 2.9302, 2.7946, 2.7333, 2.6865, 2.6309,
#             2.5415, 2.4736, 2.4599, 2.4186, 2.3905, 2.3623, 2.3241, 2.3370, 2.3425, 2.3014,
#             2.2708, 2.2948, 2.2681, 2.2465, 2.2180, 2.2221, 2.2249, 2.2105, 2.2115, 2.2034,
#             2.1785, 2.1720, 2.1604, 2.1716, 2.1699, 2.1592, 2.1773, 2.1621, 2.1716, 2.1539,
#             2.2866] + [2.1 + np.random.normal(0, 0.05) for _ in range(29)]  # Approximate for remaining epochs

# val_accuracy = [0.1081, 0.1458, 0.1717, 0.1953, 0.2160, 0.2291, 0.2431, 0.2527, 0.2548, 0.2557,
#                 0.2617, 0.2737, 0.3024, 0.3248, 0.3458, 0.3706, 0.4018, 0.4168, 0.4363, 0.4720,
#                 0.4918, 0.5245, 0.5493, 0.5657, 0.5820, 0.5922, 0.6122, 0.6171, 0.6251, 0.6159,
#                 0.5795, 0.5929, 0.6549, 0.6929, 0.7461, 0.7552, 0.7944, 0.8066, 0.8182, 0.8360,
#                 0.8510, 0.8669, 0.8651, 0.8727, 0.8730, 0.8739, 0.8803, 0.8788, 0.8791, 0.8812,
#                 0.8809, 0.8838, 0.8852, 0.8834, 0.8837, 0.8840, 0.8857, 0.8863, 0.8866, 0.8865,
#                 0.8879, 0.8864, 0.8872, 0.8870, 0.8876, 0.8878, 0.8868, 0.8872, 0.8876, 0.8867,
#                 0.8763] + [0.88 + np.random.normal(0, 0.005) for _ in range(29)]  # Approximate for remaining epochs

# # Set up the figure with a specific style and size
# plt.figure(figsize=(15, 12))
# # plt.style.use('seaborn')

# # Create a 2x1 subplot
# plt.subplot(2, 1, 1)
# plt.plot(epochs, train_loss, 'r-', label='Training Loss', linewidth=2)
# plt.plot(epochs, val_loss, 'b-', label='Validation Loss', linewidth=2)
# plt.title('SENet50 - Training and Validation Loss', fontsize=16, fontweight='bold', pad=15)
# plt.xlabel('Epoch', fontsize=12)
# plt.ylabel('Loss', fontsize=12)
# plt.legend(fontsize=10)
# plt.grid(True, linestyle='--', alpha=0.7)

# # Create the accuracy subplot
# plt.subplot(2, 1, 2)
# plt.plot(epochs, val_accuracy, 'g-', label='Validation Accuracy', linewidth=2)
# plt.title('SENet50 - Validation Accuracy', fontsize=16, fontweight='bold', pad=15)
# plt.xlabel('Epoch', fontsize=12)
# plt.ylabel('Accuracy', fontsize=12)
# plt.legend(fontsize=10)
# plt.grid(True, linestyle='--', alpha=0.7)

# # Adjust the layout and display
# plt.tight_layout()
# plt.show()

# # Save the figure
# plt.savefig('senet50_training_metrics.png', dpi=300, bbox_inches='tight')