import tkinter as tk

from PIL import Image, ImageTk
from bigram.bigram_model import *



bigram = Bigram(frequency_threshold=3)
bigram.train(train_file="data/SherlockHolmes-train.txt")
bigram.test(test_file="data/SherlockHolmes-test.txt")
model = bigram.compress()


def view_dataset():
    window2 = tk.Tk()
    window2.config(background="white")
    window2.geometry("500x500")
    window2.resizable(width=False, height=False)
    window2.title("Dataset")

    scrollbar = tk.Scrollbar(window2, orient=tk.VERTICAL)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    text_box_label = tk.Label(window2, text="SHERLOCK HOLMES", font=("ArialBold", 18), background="white")
    text_box_label.pack()
    text_box = tk.Text(window2, width=60, height=40, background="white", borderwidth=5, yscrollcommand=scrollbar.set,
                       wrap=tk.WORD, font=("TimesNewRoman", 15))

    with open("data/SherlockHolmes-train.txt", "rt", encoding="utf8") as f:
        text = f.read()

    text_box.insert(tk.END, text)

    scrollbar.config(command=text_box.yview)
    text_box.pack(pady=10)

    window2.mainloop()


def view_perplexity():
    global model
    pp = model["Perplexity"]
    label_pp.config(text="")

    label_pp.config(text=str(round(pp, 3)), font=("TimesNewRoman", 20), background="grey", borderwidth=5)
    label_pp.place(x=420, y=420)


def start_sampling():
    global model
    assert model is not None

    window3 = tk.Tk()
    window3.title("Sampling")
    window3.geometry("500x500")
    window3.config(background="white")
    window3.resizable(width=False, height=False)

    main_label = tk.Label(window3, text="GENERATED SENTENCES", font=("ArialBold", 18), background="white")
    main_label.pack()

    scrollbar = tk.Scrollbar(window3, orient=tk.VERTICAL)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    text_box = tk.Text(window3, width=50, height=50, font=("TimesNewRoman", 15), wrap=tk.WORD,
                       yscrollcommand=scrollbar.set)

    text_box.pack()

    scrollbar.config(command=text_box.yview)

    row2id, col2id = model["Row2IdMap"], model["Col2IdMap"]
    id2col = {v: k for k, v in col2id.items()}
    table = np.array(model["Table"])

    num_sentences = np.random.choice([i + 1 for i in range(20)], 1)[0]

    text = ""
    count = 1
    while count <= num_sentences:
        max_words = 12
        start = "<s>"
        row_idx = row2id[start]
        sentence = []
        while len(sentence) < max_words:
            col_idx = \
                np.random.choice([i for i in range(len(table[row_idx]))], p=[10 ** p for p in table[row_idx]], size=1)[
                    0]

            start = id2col[col_idx]
            if start == "</s>":
                break
            row_idx = row2id[start]
            sentence.append(start)

        text += f"{count}. {' '.join(sentence)}\n\n"
        count += 1

    text_box.insert(tk.END, text.strip())

    window3.mainloop()


window = tk.Tk()
window.config(background="grey", borderwidth=10)
window.geometry("1000x500")
window.resizable(width=False, height=False)
window.title("Bigram Model")

title = tk.Label(window, text="Bigram LM", font=("ALGERIAN", 35), background="grey")
title.pack(pady=10)

dataset_image = Image.open("images/view_dataset.png")
dataset_image.resize((10, 10), Image.LANCZOS)
dataset_image = ImageTk.PhotoImage(dataset_image)

dataset = tk.Button(window, text="View Dataset", image=dataset_image, background="grey", activebackground="grey",
                    command=view_dataset)
dataset["border"] = 0
dataset.place(x=50, y=100)

sampling_image = Image.open("images/start_sampling.png")
sampling_image.resize((10, 20), Image.LANCZOS)
sampling_image = ImageTk.PhotoImage(sampling_image)

sampling = tk.Button(window, text="Start Sampling", image=sampling_image, background="grey", activebackground="grey",
                     command=start_sampling)
sampling["border"] = 0
sampling.place(x=610, y=100)

perplexity = tk.Button(window, text="Perplexity", font=("ArialBold", 20), background="grey",
                       activebackground="white", command=view_perplexity)
perplexity["border"] = 0
perplexity.place(x=400, y=350)
label_pp = tk.Label()

window.mainloop()
