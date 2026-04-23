import pandas as pd
import matplotlib.pyplot as plt
from benchmark import benchmark_names

def visualize():
    plt.figure(figsize=(15, 10))
    
    # Đọc dữ liệu từ các file
    apo = pd.read_csv('apo.csv')
    aro = pd.read_csv('aro.csv')
    aoa = pd.read_csv('PA1_aoa.csv')
    coa = pd.read_csv('PA2_coa.csv')
    ga = pd.read_csv('PA3_ga.csv')
    pa4 = pd.read_csv('PA4_aro_apo.csv')
    pso = pd.read_csv('PA5_pso.csv')
    
    # Vẽ 9 đồ thị tương ứng 9 hàm
    for i, func in enumerate(benchmark_names):
        plt.subplot(2, 3, i+1)
        
        # Lọc dữ liệu theo hàm
        apo_func = apo[apo['Function'] == func]
        aro_func = aro[aro['Function'] == func]
        aoa_func = aoa[aoa['Function'] == func]
        coa_func = coa[coa['Function'] == func]
        ga_func = ga[ga['Function'] == func]
        pa4_func = pa4[pa4['Function'] == func]
        pso_func = pso[pso['Function'] == func]
        
        # Vẽ 7 đường
        plt.plot(apo_func['Iteration'], apo_func['Best_Fitness'], label='APO', color='cyan')
        plt.plot(aro_func['Iteration'], aro_func['Best_Fitness'], label='ARO', color='orange')
        plt.plot(aoa_func['Iteration'], aoa_func['Best_Fitness'], label='AOA', linestyle='--', color='green')
        plt.plot(coa_func['Iteration'], coa_func['Best_Fitness'], label='COA', linestyle='--', color='red')
        plt.plot(ga_func['Iteration'], ga_func['Best_Fitness'], label='GA', linestyle='-.', color='blue')
        plt.plot(pa4_func['Iteration'], pa4_func['Best_Fitness'], label='ARO-APO', linestyle='-.', color='black')
        plt.plot(pso_func['Iteration'], pso_func['Best_Fitness'], label='PSO', linestyle=':', color='brown')
        
        plt.title(func)
        plt.xlabel('Iteration')
        plt.ylabel('Best Fitness (log)')
        plt.yscale('log')
        plt.grid(True)
    
    # Đặt legend bên ngoài biểu đồ
    plt.tight_layout()
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), title="Algorithms")
    plt.savefig('comparison_all.png', bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    visualize()