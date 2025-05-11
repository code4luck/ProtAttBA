import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os, sys
import numpy as np
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt
import seaborn as sns


def draw_global_attn(
    attention_weights,
    antibody_sequence,
    antigen_sequence,
    mutation_idx=90,
    x_start_idx=15,
    x_end_idx=27,
    output_path=None,
):
    attention_weights = np.array(attention_weights)
    # del cls and sep
    attn_display = np.mean(attention_weights[0, :, 1:-1, 1:-1], axis=0)
    title = "Average Attention Weights"

    num_plots = 1
    fig, axs = plt.subplots(1, num_plots, figsize=(80, 66))
    if num_plots == 1:
        axs = [axs]

    ax = sns.heatmap(
        attn_display,
        cmap="Blues",
        ax=axs[0],
        cbar=True,
        xticklabels=antigen_sequence,
        yticklabels=antibody_sequence,
        square=False,
        annot=False,
        linewidths=0.5,
    )

    if mutation_idx is not None:
        axs[0].hlines(           
            y=mutation_idx,
            xmin=x_start_idx, 
            xmax=x_end_idx,  
            color="red",
            linewidth=2,
            linestyle="--",
            alpha=0.8)
        axs[0].hlines(
            y=mutation_idx + 1,
            xmin=x_start_idx, 
            xmax=x_end_idx,  
            color="red",
            linewidth=2,
            linestyle="--",
            alpha=0.8)
    axs[0].set_title(title)
    axs[0].set_xlabel("Antigen")
    axs[0].set_ylabel("Antibody")
    axs[0].tick_params(axis="x", rotation=90)
    axs[0].tick_params(axis="y", rotation=0)
    cbar = ax.collections[0].colorbar 
    cbar.ax.tick_params(labelsize=180)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.show()



def plot_row_attention_heatmap_with_numnote(
    attention_weights,
    antibody_sequence,
    antigen_sequence,
    mutation_idx=None,
    start_pos=20,
    end_pos=38,
    output_path=None,
):
    attention_weights = np.array(attention_weights)
    display_weights = np.mean(attention_weights[:, :, mutation_idx, 1:-1], axis=1)
    display_weights = display_weights[0]
    cur_AA = antibody_sequence[mutation_idx]
    plt.figure(figsize=(8, 1.5))

    display_weights = display_weights[start_pos - 1 : end_pos]
    display_sequence = antigen_sequence[start_pos - 1 : end_pos]
    ax = sns.heatmap(
        display_weights[np.newaxis, :],
        cmap="Blues",
        annot=False,
        xticklabels=display_sequence, 
        yticklabels=[f"{cur_AA}"],
        linewidths=0.5,
    )
    ax.collections[0].colorbar.remove()

    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(np.arange(len(display_sequence)) + 0.5)
    ax2.set_xticklabels(range(start_pos, end_pos + 1))
    ax.tick_params(axis="x", rotation=0, labelsize=14) 
    ax.tick_params(axis="y", rotation=0, labelsize=14)  
    ax2.tick_params(axis="x", rotation=0, labelsize=14) 
    if mutation_idx is not None and start_pos <= mutation_idx + 1 <= end_pos:
        adjusted_idx = mutation_idx - (start_pos - 1)
        plt.axvline(x=adjusted_idx, color="black", linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.show()



def load_data(data_pth):
    data = pkl.load(open(data_pth, "rb"))
    return data


if __name__ == "__main__":

    # 1IAR_global
    data_path = "./1IAR_R53Q.pkl"
    antibody_sequence = "HKCDITLQEIIKTLNSLTEQKTLCTELTVTDIFAASKNTTEKETFCRAATVLQQFYSHHEKDTRCLGATAQQFHRHKQLIRFLKRLDRNLWGLAGLNSCPVKEANQSTLENFLERLKTIMREKYSKCSS"
    antigen_sequence = "FKVLQEPTCVSDYMSISTCEWKMNGPTNCSTELRLLYQLVFLLSEAHTCIPENNGGAGCVCHLLMDDVVSADNYTLDLWAGQQLLWKGSFKPSEHVKPRAPGNLTVHTNVSDTLLLTWSNPYPPDNYLYNHLTYAVNIWSENDPADFRIYNVTYLEPSLRIAASTLKSGISYRARVRAWAQAYNTTWSEWSPSTKWHNSYREPFEQH"
    data = load_data(data_path)
    draw_global_attn(
        attention_weights=data,
        antibody_sequence=antibody_sequence,
        antigen_sequence=antigen_sequence,
        x_start_idx=35,
        x_end_idx=47,
        mutation_idx=52,
        output_path="./1IAR_global.png",
    )
    # 1IAR range
    data_path = "./1IAR_R53Q.pkl"
    antibody_sequence = "HKCDITLQEIIKTLNSLTEQKTLCTELTVTDIFAASKNTTEKETFCRAATVLQQFYSHHEKDTRCLGATAQQFHRHKQLIRFLKRLDRNLWGLAGLNSCPVKEANQSTLENFLERLKTIMREKYSKCSS"
    antigen_sequence = "FKVLQEPTCVSDYMSISTCEWKMNGPTNCSTELRLLYQLVFLLSEAHTCIPENNGGAGCVCHLLMDDVVSADNYTLDLWAGQQLLWKGSFKPSEHVKPRAPGNLTVHTNVSDTLLLTWSNPYPPDNYLYNHLTYAVNIWSENDPADFRIYNVTYLEPSLRIAASTLKSGISYRARVRAWAQAYNTTWSEWSPSTKWHNSYREPFEQH"
    data = load_data(data_path)
    plot_row_attention_heatmap_with_numnote(
        attention_weights=data,
        antibody_sequence=antibody_sequence,
        antigen_sequence=antigen_sequence,
        mutation_idx=52,
        start_pos=35,
        end_pos=47,
        output_path="./1IAR_range.png",
    )

    # 1DQJ_global
    # data_path = "./1DQJ_S91A.pkl"
    # antibody_sequence = "DIVLTQSPATLSVTPGDSVSLSCRASQSISNNLHWYQQKSHESPRLLIKYASQSISGIPSRFSGSGSGTDFTLSINSVETEDFGMYFCQQANSWPYTFGGGTKLEIKRADAAPTVSIFPPSSEQLTSGGASVVCFLNNFYPKDINVKWKIDGSERQNGVLNSWTDQDSKDSTYSMSSTLTLTKDEYERHNSYTCEATHKTSTSPIVKSFNRNECEVQLQESGPSLVKPSQTLSLTCSVTGDSVTSDYWSWIRKFPGNKLEYMGYISYSGSTYYHPSLKSRISITRDTSKNQYYLQLNSVTTEDTATYYCASWGGDVWGAGTTVTVSSAKTTAPSVYPLAPVCGDTTGSSVTLGCLVKGYFPEPVTLTWNSGSLSSGVHTFPAVLQSDLYTLSSSVTVTSSTWPSQSITCNVAHPASSTKVDKKI"
    # antigen_sequence = "KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTPGSRNLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL"
    # data = load_data(data_path)
    # draw_global_attn(
    #     attention_weights=data,
    #     antibody_sequence=antibody_sequence,
    #     antigen_sequence=antigen_sequence,
    #     mutation_idx=90,
    #     x_start_idx=15,
    #     x_end_idx=27,
    #     output_path="./1DQJ_global.png",
    # )
    # 1DQJ_range
    # data_path = "./1DQJ_S91A.pkl"
    # antibody_sequence = "DIVLTQSPATLSVTPGDSVSLSCRASQSISNNLHWYQQKSHESPRLLIKYASQSISGIPSRFSGSGSGTDFTLSINSVETEDFGMYFCQQANSWPYTFGGGTKLEIKRADAAPTVSIFPPSSEQLTSGGASVVCFLNNFYPKDINVKWKIDGSERQNGVLNSWTDQDSKDSTYSMSSTLTLTKDEYERHNSYTCEATHKTSTSPIVKSFNRNECEVQLQESGPSLVKPSQTLSLTCSVTGDSVTSDYWSWIRKFPGNKLEYMGYISYSGSTYYHPSLKSRISITRDTSKNQYYLQLNSVTTEDTATYYCASWGGDVWGAGTTVTVSSAKTTAPSVYPLAPVCGDTTGSSVTLGCLVKGYFPEPVTLTWNSGSLSSGVHTFPAVLQSDLYTLSSSVTVTSSTWPSQSITCNVAHPASSTKVDKKI"
    # antigen_sequence = "KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTPGSRNLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL"
    # data = load_data(data_path)
    # plot_row_attention_heatmap_with_numnote(
    #     attention_weights=data,
    #     antibody_sequence=antibody_sequence,
    #     antigen_sequence=antigen_sequence,
    #     mutation_idx=90,
    #     start_pos=15,
    #     end_pos=27,
    #     output_path="./1DQJ_range.png",
    # )
