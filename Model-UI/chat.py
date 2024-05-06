import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
import streamlit as st
from streamlit import session_state

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json','r') as json_data:
  intents = json.load(json_data)
  
FILE = "data.pth"
data = torch.load(FILE)

if "listamessaggi" not in session_state:
    session_state.listamessaggi = []

st.set_page_config(page_title="ChatBot - AI", page_icon="robot.png", layout="centered", initial_sidebar_state="expanded")

gradient_text_html = """
  <style>
  .gradient-text {
      font-weight: bold;
      background: rgb(190,247,255);
      background: linear-gradient(90deg, rgba(190,247,255,1) 0%, rgba(65,171,255,1) 71%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      display: inline;
      font-size: 3em;
  }
  </style>
  <div class="gradient-text">Assistente Virtuale - AI Chatbot</div>
  """

st.markdown(gradient_text_html, unsafe_allow_html=True)
st.caption("L'assistente virtuale per rispondere alle tue domande sui problemi del tuo piano telefonico, attivo 24/7!")

user_question = st.chat_input("Fai la domanda:")



input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Michele"

with st.sidebar:
    st.title('💬 AI ChatBot')
    st.write('Questo chatbot è stato addestrato su un dataset di domande e risposte. Prova a fare una domanda! 🤖')
    st.write('E stato creato tramite la libreria PyTorch per il modello di machine learning e rappresentato graficamente con il framework Streamlit.')
    st.write('Credits: ')
    ul_list = """
      <ul>
        <li>Niccolò Cetoloni</li>
        <li>Lorenzo Lombrichi</li>
        <li>Mirko Bruttini</li>
      </ul>
      """

    # Visualizzazione della lista non ordinata utilizzando st.write() con HTML
    st.write(ul_list, unsafe_allow_html=True)
    st.write('Docente: Simone Giuliani')
    st.write('Istituto Tecnico Industriale Tito Sarrocchi - Siena')
    st.write('Anno scolastico 2023/2024')
    st.divider()
    st.title("📂 Repository")
    st.write("📚 [GitHub Repository](https://github.com/Nicco-2603/ChatBotAI---Project)")

def clear_chat_history():
    st.session_state.listamessaggi = []
st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

colore = "#0D161D"
icona = ""
colore_bordo =""
direzione_ombra =""
divs = ""
listamessaggi = []
if user_question:
    response = "You: " + user_question

    sentence = user_question
    session_state.listamessaggi.append(response)
    response_text = ""
    while True:
          sentence = tokenize(sentence)
          X = bag_of_words(sentence, all_words)
          X = X.reshape(1,X.shape[0])
          X = torch.from_numpy(X).to(device)
          
          output = model(X)
          _, predicted = torch.max(output, dim=1)
          tag = tags[predicted.item()]
          
          probs = torch.softmax(output, dim=1)
          prob = probs[0][predicted.item()]
          
          if prob.item() > 0.75:
            for intent in intents['intents']:
              if tag == intent["tag"]:
                risp = random.choice(intent['responses'])  
                print(f"{bot_name}: {risp}")
                response_text = f"{bot_name}: {risp}"
                session_state.listamessaggi.append(response_text)
                print(session_state.listamessaggi)
                for messaggio in session_state.listamessaggi:
                    if messaggio.startswith("You:"):
                        colore = "#002949"
                        colore_bordo = "#c8e7ff"
                        icona="\Model-UI\img\robot.png"
                        
                    else:
                        colore = "#001a2e"
                        colore_bordo = "#41ABFF"
                        direzione_ombra="3px"
                        icona = "https://media.discordapp.net/attachments/777256549302403142/1235613698580545618/icon3.png?ex=6635026e&is=6633b0ee&hm=04147b16256c74397b720999380a9e9e8706b082a9dc4a9dbdd92acbc2d44ab0&=&format=webp&quality=lossless"
                    divs += f"""

                        <div style="border:1px solid black;
                        box-shadow: 3px 3px 0px {colore_bordo};
                        padding:10px; 
                        margin: 5px auto 20px auto; 
                        background-color:{colore};
                        border-radius:10px; 
                        border-color:{colore_bordo}; 
                        text-align:left;
                        width:auto;"
                        >   
                            <img src="{icona}" width="45" height="40">
                            {messaggio}
                        </div>
                    """
                st.markdown(divs, unsafe_allow_html=True)
                               
          else:
            print(f"{bot_name}: Non ho capito, riprova.")
            colore = "#001a2e"
            colore_bordo = "#41ABFF"
            icona = "https://media.discordapp.net/attachments/777256549302403142/1235613698580545618/icon3.png?ex=6635026e&is=6633b0ee&hm=04147b16256c74397b720999380a9e9e8706b082a9dc4a9dbdd92acbc2d44ab0&=&format=webp&quality=lossless"
            divs += f"""
                        <div style="border:1px solid black;
                        box-shadow: 3px 3px 0px {colore_bordo};
                        padding:10px; 
                        margin: 5px auto 20px auto; 
                        background-color:{colore};
                        border-radius:10px; 
                        border-color:{colore_bordo}; 
                        text-align:left;
                        width:auto;"
                        >   
                            <img src="{icona}" width="45" height="40">
                            {bot_name}: Non ho capito, riprova.
                        </div>
                    """
            st.markdown(divs, unsafe_allow_html=True)
          break