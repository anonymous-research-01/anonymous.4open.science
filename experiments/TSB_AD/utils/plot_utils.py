import matplotlib.pyplot as plt

def range_convers_new(label):
    '''
    input: arrays of binary values
    output: list of ordered pair [[a0,b0], [a1,b1]... ] of the inputs
    '''
    L = []
    i = 0
    j = 0
    while j < len(label):
        # print(i)
        while label[i] == 0:
            i+=1
            if i >= len(label):
                break
        j = i+1
        # print('j'+str(j))
        if j >= len(label):
            if j==len(label):
                L.append((i,j-1))

            break
        while label[j] != 0:
            j+=1
            if j >= len(label):
                L.append((i,j-1))
                break
        if j >= len(label):
            break
        L.append((i, j-1))
        i = j
    return L


def plotFigures_systhetic_data(label_ranges,label_array_list,method_name_list,file_name=None,save_dir=None,slidingWindow=100,
                               color_box=0.2, plotRange=None, save_plot=False,
                               plot_1_name='Real Data', plot_2_name='Perfect Model', plot_3_name='Model 1 (MVN)',
                               plot_4_name='Model 2 (AE)', plot_5_name='Random Score',):
    range_anomaly = label_ranges[0]
    # range_anomaly = range_convers_new(label_ranges[0])


    score = label_array_list[0]

    max_length = len(score)
    if plotRange is None:
        plotRange = [0, max_length]

    # fig3 = plt.figure(figsize=(10, 5), constrained_layout=True)
    fig3 = plt.figure(figsize=(20, 10), constrained_layout=True)
    gs = fig3.add_gridspec(len(label_array_list), 1)  # Adjusted grid for 5 rows

    # Function to plot each anomaly score
    def plot_anomaly_score(ax, score, label_text,idx):
        ax.plot(score[:max_length])

        # for choose anomaly
        # if idx != 0:
        #     for r in range_anomaly:
        #         ax.axvspan(r[0], r[1], color='red', alpha=color_box)
        # else:
        #     for r in range_anomaly:
        #         ax.axvline(x=r[0], color='gray', linestyle='--', linewidth=1, alpha=0.7)
        #         ax.axvline(x=r[1], color='gray', linestyle='--', linewidth=1, alpha=0.7)

        # for choose anomaly and pred
        for r in range_anomaly:
            ax.axvspan(r[0], r[1], color='red', alpha=color_box)

        ax.set_ylabel('score')
        ax.set_xlim(plotRange)
        # ax.text(0.02, 0.90, label_text, transform=ax.transAxes, fontsize=12, verticalalignment='top',
        #         bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='white'))
        ax.text(0.02, 0.90, label_text, transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='white', alpha=0))


    # Plotting the anomaly scores in separate subplots
    for i, label_array in enumerate(label_array_list):
        f3_ax1 = fig3.add_subplot(gs[i, 0])
        # left_msg = "pred"+i.__str__() if i !=0  else "gt"
        left_msg = method_name_list[i] if i !=0  else file_name
        plot_anomaly_score(f3_ax1, label_array, left_msg,i)


    if save_dir != None and file_name!= None:
        plt.savefig(save_dir+file_name.split(".")[0]+".png")  # 保存为PNG文件

    plt.show()

    return fig3

def plotFigures_systhetic_data1(label_ranges,label_array_list,method_name_list,file_name=None,save_dir=None,slidingWindow=100,
                               color_box=0.2, plotRange=None, show_pic = True,save_plot=False,
                               plot_1_name='Real Data', plot_2_name='Perfect Model', plot_3_name='Model 1 (MVN)',
                               plot_4_name='Model 2 (AE)', plot_5_name='Random Score',):
    range_anomaly = label_ranges[0]
    # range_anomaly = range_convers_new(label_ranges[0])


    score = label_array_list[0]

    max_length = len(score)
    if plotRange is None:
        plotRange = [0, max_length]

    # fig3 = plt.figure(figsize=(10, 5), constrained_layout=True)
    fig3 = plt.figure(figsize=(20, 10), constrained_layout=True)
    gs = fig3.add_gridspec(len(label_array_list), 1)  # Adjusted grid for 5 rows

    # Function to plot each anomaly score
    def plot_anomaly_score(ax, score, label_text,idx):
        ax.plot(score[:max_length])

        # for choose anomaly
        # if idx != 0:
        #     for r in range_anomaly:
        #         ax.axvspan(r[0], r[1], color='red', alpha=color_box)
        # else:
        #     for r in range_anomaly:
        #         ax.axvline(x=r[0], color='gray', linestyle='--', linewidth=1, alpha=0.7)
        #         ax.axvline(x=r[1], color='gray', linestyle='--', linewidth=1, alpha=0.7)

        # for choose anomaly and pred
        for r in range_anomaly:
            ax.axvspan(r[0], r[1], color='red', alpha=color_box)

        ax.set_ylabel('score')
        ax.set_xlim(plotRange)
        # ax.text(0.02, 0.90, label_text, transform=ax.transAxes, fontsize=12, verticalalignment='top',
        #         bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='white'))
        ax.text(0.02, 0.90, label_text, transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='white', alpha=0))


    # Plotting the anomaly scores in separate subplots
    for i, label_array in enumerate(label_array_list):
        f3_ax1 = fig3.add_subplot(gs[i, 0])
        # left_msg = "pred"+i.__str__() if i !=0  else "gt"
        # left_msg = method_name_list[i] if i !=0  else file_name
        left_msg = method_name_list[i]
        plot_anomaly_score(f3_ax1, label_array, left_msg,i)


    if save_dir != None and file_name!= None:
        # plt.savefig(save_dir+file_name.split(".")[0]+".png")  # 保存为PNG文件
        root_path = "D:/projects/metric/PATE/experiments/Synthetic_Data_Experiments"
        # save_path_svg = root_path+"/paper/src/figures/" + "labeling_problem_case_"+file_name + ".svg"
        save_path_pdf = root_path+"/paper/src/figures/" + "labeling_better_case_"+file_name + ".pdf"
        save_path_png = root_path+"/paper/src/figures/" + "labeling_better_case_"+file_name + ".png"
        # plt.savefig(save_path_svg, format='svg')
        # 保存为 PDF 文件
        plt.savefig(save_path_pdf, format="pdf", bbox_inches="tight")
        plt.savefig(save_path_png)
        print("pic create finish!")

    if show_pic:
        plt.show()

    return fig3