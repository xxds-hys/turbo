import matplotlib.pyplot as plt
import numpy as np

#llama的数据
tokens = ['512', '768', '1024', '1280', '1536', '1792', '2048']
turbo_data = [0.004751, 0.004359, 0.004209, 0.003931, 0.004546, 0.004997, 0.005067]
int8_data = [0.003955, 0.002925, 0.002762, 0.002877, 0.003168, 0.003407, 0.004514]

#gpt的数据
'''
tokens = ['512', '768', '1024']
turbo_data = [0.000132, 0.000087, 0.000067]
int8_data = [0.000232, 0.000159, 0.000122]
'''
#qwen的数据
'''
tokens = ['512', '768', '1024', '1280', '1536', '1792', '2048']
turbo_data = [0.012496, 0.007767, 0.005711, 0.00501, 0.004933, 0.0052, 0.005898]
int8_data = [0.024954, 0.016498, 0.012746, 0.011167, 0.010572, 0.016689, 0.021613]
'''
x = np.arange(len(tokens))
width = 0.35

plt.figure(figsize=(12, 7), dpi=100)


rects1 = plt.bar(x - width/2, turbo_data, width, label='Turbo', color='#1f77b4')
rects2 = plt.bar(x + width/2, int8_data, width, label='Int8', color='#ff7f0e')


plt.axhline(y=0.05, color='r', linestyle='--', linewidth=1.5, alpha=0.8, label='Threshold (0.05)')

plt.xlabel('Token Count', fontsize=32)
plt.ylabel('KL Divergence', fontsize=32)
#plt.title('Llama 2 7B', fontsize=32, fontweight='bold')
plt.xticks(x, tokens, fontsize=32)
plt.yticks(fontsize=30)
plt.legend(loc='upper right', bbox_to_anchor=(1.0, 0.9), fontsize=30, frameon=True)

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        plt.annotate(f'{height:.4f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=11)

plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
#plt.savefig('klllama.png', dpi=300, bbox_inches='tight')
plt.show()