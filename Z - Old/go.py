from transformers import BertTokenizer, BertForQuestionAnswering

# pip freeze > requirements.txt
# pip install -r requirements.txt
# Load the BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('fabgraziano/bert-italian-xxl-cased_squad-it_v1')
model = BertForQuestionAnswering.from_pretrained('fabgraziano/bert-italian-xxl-cased_squad-it_v1')

text = "Ciao, mi chiamo Mike, sono di Milano e sono uno studente di informatica. Mi piacciono i gatti e giocare a calcio. In realtà il mio linguaggio preferito è Javascript ma adoro anche Java"

# Cerca l'articolo contenente la parola chiave

while (True):
    # Define the input text and question
    
    print(text)
    print("\n\n")
    print("Scrivi la domanda in inglese: ")
    
    question = input()
    #"Who was revealed to have been embezzling funds from the company?"

    # Encode the input text and question, and get the scores for each word in the text
    prompt = tokenizer(question, text,  return_tensors="pt")

    output = model(**prompt)

    # Find the words in the text that corresponds to the highest start and end scores
    # with a torch.no_grad():
    #     outputs = model(**inputs)
    start_index = output.start_logits.argmax()
    end_index = output.end_logits.argmax() + 1

    # Extract the span of words as the answer
    answer = tokenizer.decode(prompt.input_ids[0, start_index:end_index])
    print(answer)