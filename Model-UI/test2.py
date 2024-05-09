import matplotlib.pyplot as plt

losses=[ 0.0105, 0.0240, 0.0019, 0.0304, 0.4642,0.0001,0.0000,0.0000]
# Crea un array per gli epoch
epochs = range(1, len(losses) + 1)

# Plotta le losses
plt.plot(epochs, losses, 'b', label='Training loss')
plt.title('Training loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()