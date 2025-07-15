import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import os
import csv
from datetime import datetime
from Tools import get_explanation,load_data,similarity_checker,get_confidence,load_toEndlessTuning,DecisionTree_Tuning,prepare_tuning,inverse_relabeler

def go_to_page(pagina_dest):
    st.session_state.pagina = pagina_dest
 
def log_interaction(interaction_type, details=f""):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open('user_interactions.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([timestamp, interaction_type, details])
    success = 'OK'
    print('Log_interaction_called')
    return success


####    INTRO   ###############################
def pagina_1():
    if os.path.exists('./user_interactions.csv'):
        os.remove('./user_interactions.csv')

    st.markdown("<h1 style='text-align: center;'>The Endless Tuning</h1>", unsafe_allow_html=True) 

    # Box informativo
    st.markdown("""
        <div style="background-color: #000000; border-radius: 1vw; padding: 2vw; margin-top: 2vw;">
            <p style="font-size: 100%; text-align: justify;">Welcome! This is the Endless Tuning, a relational approach to artificial intelligence based on
                a continuous interaction and a double-mirror feedback. Here, in particular, in a decision-making setting for
                classification. Firstly, you will be asked to load a case. If you will feel confident about the case
                you will be simply asked of providing the right label. If otherwise you will feel unconfident,
                some suggestions and clues coming from the machine will nudge you to reflect. At the end of the process, you will have the last word.
                After a sufficient number of sessions, the model will automatically update its parameters, 
                so as to "tune" with your own style and learn together with you. You don't have any time constraint</p>
                <p style="font-size: 100%; text-align: justify;"> Note that <strong>every</strong> interaction will be recorded.
                However, records will be totaly anonymous, deleted at the end of every cycle. In the end, you will be able to download your data.
                By clicking on the "I understand" button, you declare that you understand and agree.</p>
        </div>
    """, unsafe_allow_html=True)
    st.write('')
    cola,colb = st.columns([0.7,1.05])
    with colb:
        if st.button("I understand"):
            log_interaction("pressed key I understand")
            go_to_page(2) 
            st.rerun()


####    LOADING DATABASE   ##################################
def pagina_2():

    if 'notes3' in st.session_state:
        del st.session_state.notes3
    if 'notes4' in st.session_state:
        del st.session_state.notes4
    if 'notes5' in st.session_state:
        del st.session_state.notes5
    if 'notes6' in st.session_state:
        del st.session_state.notes6

    st.title("setting #1: loan granting")
    st.write("""You are a bank manager and today someone has come to you, applying for a loan.
    You have access to much information about the applicant. According to your expertise and, 
    if you prefer, also to artificial intelligence's suggestions, your task is to decide whether 
    the application is worth the granting or not. In order to proceed, select a database and 
    a loan applicant ID.""")
    

    #db = st.file_uploader(".csv database:", type = ["csv"])
    dbpath = './dataset/Case_Studies.csv'
    db = pd.read_csv(dbpath)
    if db is not None:
        st.session_state.db = db
        st.session_state.dbpath = dbpath #'./dataset/'+db.name
        #database = pd.read_csv(db)
        database = pd.DataFrame(db)#atabase) 
        st.session_state.database = database
        ID = st.text_input(f"Key index of the loan applicant. #ID must be an integer number comprised between 0 and {len(database)}") 
        
        if ID is not "" and ID.isdigit() and int(ID) <= len(database):
            st.session_state.ID = ID
            if 'log1' not in st.session_state:
                    st.session_state.log1 = log_interaction(f"Loaded case idx: {ID} {load_toEndlessTuning(idx=ID,data_path=st.session_state.dbpath)}")

    cola,colb = st.columns([6.6,1])
    with cola:
        if st.button("Back"):
            if 'log2' not in st.session_state:
                st.session_state.log2 = log_interaction("Going back...")
            go_to_page(1)
            st.rerun()  
    with colb:
        proceed_button = st.button('Proceed')
    if proceed_button:
        if db is None:
            log_interaction("Error: tried to proceed without selecting a file.")
            st.markdown(f"<p style='font-size: 1vw;'>Error: select a valid file before proceeding!</p>", unsafe_allow_html=True)
        elif ID is "" or not ID.isdigit():
            log_interaction("Error: tried to proceed without selecting a valid applicant.")
            st.markdown(f"<p style='font-size: 1vw;'>Error: select a valid applicant ID before proceeding!</p>", unsafe_allow_html=True)
        elif ID.isdigit() and not int(ID) <= len(database):
            log_interaction("Error: tried to proceed without selecting a valid applicant.")
            st.markdown(f"<p style='font-size: 1vw;'>Error: select a valid applicant ID before proceeding!</p>", unsafe_allow_html=True)
        else:
            if 'log3' not in st.session_state:
                st.session_state.log3 = log_interaction("Proceeding to next page...")
            go_to_page(3)  
            st.rerun()  


####    FIRST IMPRESSION    #################################
def pagina_3():
    if 'log1' in st.session_state:
        del st.session_state.log1
    if 'log2' in st.session_state:
        del st.session_state.log2
    if 'log3' in st.session_state:
        del st.session_state.log3

    st.title("What's your impression?")
    st.write("""Speak your mind! Here you can find the applicant's features.
    Take your time and select the purported class. You may decide whether to 
    go on or to stop here and go to the next session. If you feel confident, 
    press 'Save and exit'. Otherwise, if you think you could receive useful 
    suggestions from artificial intelligence, please proceed.""")
    st.markdown("<br>", unsafe_allow_html=True) 
    
    if "ID" in st.session_state:
        applicant = inverse_relabeler(st.session_state.ID,st.session_state.dbpath)
        diagram = {}
        for i in range(len(applicant.columns)):
            diagram[applicant.columns[i]] = applicant.iloc[0][i]
        
        st.markdown(
            """
            <style>
            .custom-table {
                width: 100%;
                border-collapse: collapse;
            }
            .custom-table th, .custom-table td {
                border: 1px solid white;
                padding: 10px;
                text-align: left;
                color: sandybrown;
                font-size: 16px;
            }
            .custom-table th {
                background-color: transparent;
                color: skyblue;
            }
            .custom-table td {
                background-color: transparent;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        # Creazione della tabella HTML con i dati
        table_html = "<table class='custom-table'>"
        for key, value in diagram.items():
            table_html += f"<tr><th>{key}</th><td>{value}</td></tr>"
        table_html += "</table>"

        # Mostrare la tabella in Streamlit
        st.markdown(table_html, unsafe_allow_html=True)
    
        if 'notes3' not in st.session_state:
            st.session_state.notes3 = "" 
        st.markdown("<br>", unsafe_allow_html=True) 
        st.session_state.notes3 = st.text_area("Add notes. They'll be recorded.",
                                                value=st.session_state.notes3, height=70)
    
    selected_class = st.selectbox("Select a class:", options)
    st.markdown("<br>", unsafe_allow_html=True) 

    cola,colb,colc = st.columns([4.5,5.25,1.5])
    with cola:
        if st.button("Back"):
            if 'log4' not in st.session_state:
                st.session_state.log4 = log_interaction("temporary_notes_3", st.session_state.notes3) 
            if 'log5' not in st.session_state:
                st.session_state.log5 = log_interaction(f'First impression: {selected_class}')
            log_interaction("Going back...")  
            del st.session_state.ID
            go_to_page(2)
            st.rerun()  
    with colb:
        save_pthree = st.button('Save and exit')
    if save_pthree: 
        if selected_class != 'None':
    
            if 'log4' not in st.session_state:
                st.session_state.log4 = log_interaction("temporary_notes_3", st.session_state.notes3) 
            if 'log5' not in st.session_state:
                st.session_state.log5 = log_interaction(f'First impression: {selected_class}')

            prepare_tuning(loaded=load_toEndlessTuning(idx=int(st.session_state.ID),data_path=st.session_state.dbpath),human_label=int(class_dict[selected_class]))


            go_to_page(10)
            log_interaction("Saved their impression at page 3", st.session_state.notes3)  
            st.rerun()
        else:
            st.markdown(f"<p style='font-size: 1vw;'>Error: select a class before proceeding!</p>", unsafe_allow_html=True)
    with colc:
        proceed_pthree = st.button("Proceed")
    if proceed_pthree:    
        if selected_class != 'None':

            if 'log4' not in st.session_state:
                st.session_state.log4 = log_interaction("temporary_notes_3", st.session_state.notes3) 
            if 'log5' not in st.session_state:
                st.session_state.log5 = log_interaction(f'First impression: {selected_class}')
            go_to_page(4)
    
            st.rerun()
        else:
            st.markdown(f"<p style='font-size: 1vw;'>Error: select a class before proceeding!</p>", unsafe_allow_html=True)

 
####    HINTS   ####################################
## EXPLANATION ##
def pagina_4():        
    if 'log4' in st.session_state:
        del st.session_state.log4
    if 'log5' in st.session_state: 
        del st.session_state.log5

    st.title("Some clues")
    st.write("""Some information about the applicant's features has been 
    extracted from the decision tree algorithm, according to split conditions. 
    It will appear in a various number of text boxes. It is both about putting 
    in evidence some properties and providing a possible reference to interpret 
    them. On your right, you find a table of the actual features of the applicant.
    Take your time, then update or confirm your hypothesis.
             
    Warning: the model might have paid attention to irrelevant factors.""")


    col2,col1 = st.columns([0.3,0.25])

    with col1:
        if "ID" in st.session_state:
            applicant = inverse_relabeler(st.session_state.ID,st.session_state.dbpath)
            diagram = {}
            for i in range(len(applicant.columns)):
                diagram[applicant.columns[i]] = applicant.iloc[0][i]
            
            st.markdown(
                """
                <style>
                .custom-table {
                    width: 100%;
                    border-collapse: collapse;
                }
                .custom-table th, .custom-table td {
                    border: 1px solid white;
                    padding: 15%px;
                    text-align: left;
                    color: white;
                    font-size: 90%;
                }
                .custom-table th {
                    background-color: transparent;
                }
                .custom-table td {
                    background-color: transparent;
                }
                </style>
                """,
                unsafe_allow_html=True
            )

            # Creazione della tabella HTML con i dati
            table_html = "<table class='custom-table'>"
            for key, value in diagram.items():
                table_html += f"<tr><th>{key}</th><td>{value}</td></tr>"
            table_html += "</table>"

            # Mostrare la tabella in Streamlit
            st.markdown(f'<div style="width: 100%;">{table_html}</div>', unsafe_allow_html=True)
    
    with col2:
        st.write('')
        st.write('Just as stimuli for reflection, it might be worth noting that:')
        if 'exp' not in st.session_state:

            print('Extracting local explanation...')
            rule = get_explanation(idx=int(st.session_state.ID),
                            data_path=st.session_state.dbpath)
            
            colors = ['sandybrown','aqua','yellow','violet']
            for i,colour in zip(rule,colors):
                st.markdown(f"""<div style="font-size:125%; color:{colour};font-family: 'Courier New';",
                            <p><strong>- {i}</strong></p>
                            </div>""",unsafe_allow_html=True)
            st.session_state.exp = rule
            log_interaction('Suggestions: ',st.session_state.exp)
    
        st.write('')
        st.write('Your previous annotations:')
        st.markdown(f"<h3 style='font-size: 0.8vw;'>{st.session_state.notes3}</h3>", unsafe_allow_html=True)
        
        if 'notes4' not in st.session_state:
            st.session_state.notes4 = "" 
        st.session_state.notes4 = st.text_area("Add notes.They'll be recorded.",
                                                value=st.session_state.notes4, height=100)
        selected_class = st.selectbox("Select a class:", options)


    cola, colb = st.columns([10, 1.5])
    with cola:
        if st.button("Back"):
            if 'log6' not in st.session_state:
                st.session_state.log6 = log_interaction("temporary_notes_4", st.session_state.notes4) 
            if 'log7' not in st.session_state:
                st.session_state.log7 = log_interaction(f'Second impression: {selected_class}')
            log_interaction("Going back...")  
            del st.session_state.exp
            go_to_page(3)
            st.rerun()  
    #with colb: 
        proceed_pfour = st.button("Proceed")
        if proceed_pfour: 
            if selected_class != 'None':
                if 'log6' not in st.session_state:
                    st.session_state.log6 = log_interaction("temporary_notes_4", st.session_state.notes4) 
                if 'log7' not in st.session_state:
                    st.session_state.log7 = log_interaction(f'Second impression: {selected_class}')
                go_to_page(5)
                st.rerun()
            else:
                st.markdown(f"<p style='font-size: 1vw;'>Error: select a class before proceeding!</p>", unsafe_allow_html=True)


####    SIMILARITY  ############################################
def pagina_5():

    if 'log6' in st.session_state:
        del st.session_state.log6
    if 'log7' in st.session_state:
        del st.session_state.log7

    st.title("Similarity")
    st.write("""Here you can find (from left to right) the 3 most similar cases, 
    taken from the original dataset, according to compressed representations obtained through 
    Principal Component Analysis.""")

    x_values, y_values, cs_x, cs_y, s_inst = similarity_checker(
        idx=int(st.session_state.ID),
        data_path=st.session_state.dbpath,
        origin='./dataset/Training_Data.csv',
        num_points=3
    )

    
    applicant = inverse_relabeler(st.session_state.ID,st.session_state.dbpath)
    applicant['Loan_Status'] = ''
    ap_t = applicant.T
    ap_t = ap_t.rename(columns=lambda x: 'CASE STUDY ')

    sit = s_inst.T
    sit = sit.rename(columns=lambda x: 'ID '+str(x)) 
    for i in range(len(sit.columns)):
        if sit.iloc[-1][i] == 0 or sit.iloc[-1][i] == '0':
            sit.iloc[-1][i] = 'Denied'
        elif sit.iloc[-1][i] == 1 or sit.iloc[-1][i] == '1':
            sit.iloc[-1][i] = 'Granted' 
    sit = pd.concat([sit,ap_t],axis=1)
    html_table = sit.to_html(classes="custom-table", escape=False)
    st.markdown(
        """
        <style>
        .custom-table {
            border-collapse: collapse;
            width: 100%;

            color: white;
            background: transparent;
            font-size: 75   %;
        }
        .custom-table th, .custom-table td {
            border: 1px solid white;
            padding: 8px;
            text-align: center;
            color: deepskyblue;
        }

        .custom-table td {
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.markdown(f'''
        <div style="max-width: 100vw; overflow-x: auto;">
            {html_table}
        </div>
    ''', unsafe_allow_html=True)

    st.write('')
    st.write('')
    st.write("""The following plot could help you to spatially 
    visualize the relative position. Zoom + or - if necessary, 
    hover on the points to get some informaion, then update 
    or confirm your hypothesis. """)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[cs_x],  
        y=[cs_y],  
        mode='markers+text',
        marker=dict(size=25, color='yellow', symbol='circle'),
        text=['APPLICANT'],
        textposition='top right',  
        textfont=dict(color='white', size=20),
        showlegend=False
    ))
    
    cls_dict = {'0':'DENIED','1':'GRANTED'}
    cls_dplot = []
    idx_dplot = []
    cls_gplot = []
    idx_gplot = []
    for i in range(len(s_inst)):
        
        if s_inst.iloc[i][-1] == 0:
            cls_dplot.append(str(s_inst.iloc[i][-1]))
            idx_dplot.append(s_inst.index[i])
        else:
            cls_gplot.append(str(s_inst.iloc[i][-1]))
            idx_gplot.append(s_inst.index[i])

    for i in range(len(s_inst)):
        fig.add_trace(go.Scatter(
            x = [x_values[s_inst.index.get_loc(idx)] for idx in idx_dplot],
            y=[y_values[s_inst.index.get_loc(idx)] for idx in idx_dplot],
            mode='markers+text',
            marker=dict(size=25, color='red'),
            text=[f"ID {idx}: {cls_dict[cls]}" for idx,cls in zip(idx_dplot,cls_dplot)],
            textposition="top center",  # Mantieni il testo in cima
            textfont=dict(size=20, color="white"),
            showlegend=False
        ))              

    for i in range(len(s_inst)):
        fig.add_trace(go.Scatter(
            x = [x_values[s_inst.index.get_loc(idx)] for idx in idx_gplot],
            y=[y_values[s_inst.index.get_loc(idx)] for idx in idx_gplot],
            mode='markers+text',
            marker=dict(size=25, color='green'),
            text=[f"ID {idx}: {cls_dict[cls]}" for idx,cls in zip(idx_gplot,cls_gplot)],
            textposition="top center",  # Mantieni il testo in cima
            textfont=dict(size=20, color="white"),
            showlegend=False
        ))  

    x_center = (min(x_values) + max(x_values)) / 2
    y_center = (min(y_values) + max(y_values)) / 2
    point_pad = 0.03

    # Impostiamo i margini e il range per tenere i punti al centro
    fig.update_layout(
        margin=dict(t=20, b=20, l=200, r=200),  # Aggiungi margini per evitare tagli
        width=200,  # Imposta la larghezza esplicitamente
        height=300,  # Imposta l'altezza esplicitamente
        xaxis=dict(
            range=[x_center - point_pad, x_center + point_pad],  # Aggiungi margini intorno al centro
        ),
        yaxis=dict(
            range=[y_center - point_pad, y_center + point_pad],  # Aggiungi margini intorno al centro
        )
    )

    st.plotly_chart(fig, use_container_width=True)
        
    st.write('')
    st.write('Your previous annotations:')
    st.markdown(f"<h3 style='font-size: 0.8vw;'>{st.session_state.notes3}</h3>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='font-size: 0.8vw;'>{st.session_state.notes4}</h3>", unsafe_allow_html=True)

    if 'notes5' not in st.session_state:
        st.session_state.notes5 = "" 
    st.session_state.notes5 = st.text_area("Add notes. They'll be recorded.",
                                            value=st.session_state.notes5, height=100)
    selected_class = st.selectbox("Select a class:", options)

    cola, colb = st.columns([10, 1.5])
    with cola:
        if st.button("Back"):
            if 'log8' not in st.session_state:
                st.session_state.log8 = log_interaction("temporary_notes_5", st.session_state.notes5) 
            if 'log9' not in st.session_state:
                st.session_state.log9 = log_interaction(f'Third impression: {selected_class}')
            log_interaction("Going back...")  
            go_to_page(4)
            st.rerun()  
    with colb:
        proceed_pfive = st.button('Proceed')
    if proceed_pfive:
        if selected_class != 'None':
            if 'log8' not in st.session_state:
                st.session_state.log8 = log_interaction("temporary_notes_5", st.session_state.notes5) 
            if 'log9' not in st.session_state:
                st.session_state.log9 = log_interaction(f'Third impression: {selected_class}')
            go_to_page(6)
            st.rerun()
        else:
            st.markdown(f"<p style='font-size: 1vw;'>Error: select a class before proceeding!</p>", unsafe_allow_html=True)


####    CONFIDENCES ###########################################
def pagina_6():
    if 'log8' in st.session_state:
        del st.session_state.log8
    if 'log9' in st.session_state:
        del st.session_state.log9

    st.title("Confidence")
    st.write("""Last step! You're observing the output of the decision
    tree classifier. The histogram represents the probabilities for 
    each class. Hover on the bars to get more precise information. 
    If you feel undecided, press 'I will choose later' and 
    nothing will happen.
             
    Warning: the model's accuracy is no 100%,\
    and it might be overconfident!""")
    
    
    col2, col1 = st.columns([1,1])
    
    with col2:
        st.write('')
        confidence,classes =  get_confidence(st.session_state.ID,st.session_state.dbpath)   
        classes = [str(i) for i in list(classes)]
    
        fig = go.Figure(data=[go.Bar(
            x=[label_dict[i] for i in classes],  # Le etichette per l'asse X
            y=confidence[0], # Le probabilitÃƒÂ  sui bin
            marker=dict(color='skyblue', line=dict(color='black', width=1)),
            opacity=0.75
        )])

        fig.update_layout(
            #title="Histogram of probabilities",
            yaxis_title="Probability",
            template="plotly_dark",  # Tema scuro
            plot_bgcolor="rgb(30, 30, 30)",  # Sfondo scuro
            paper_bgcolor="rgb(20, 20, 20)",  # Sfondo del grafico
            font=dict(color='white'),  # Colore del testo
            xaxis_tickangle=-45,  # Angolo delle etichette sull'asse X
            yaxis=dict(range=[0,1])
        )

        st.plotly_chart(fig,use_container_width=True)
    
    with col1:
        if "ID" in st.session_state:
            applicant = inverse_relabeler(st.session_state.ID,st.session_state.dbpath)
            diagram = {}
            for i in range(len(applicant.columns)):
                diagram[applicant.columns[i]] = applicant.iloc[0][i]
            
            st.markdown(
                """
                <style>
                .custom-table {
                    width: 100%;
                    border-collapse: collapse;
                }
                .custom-table th, .custom-table td {
                    border: 1px solid white;
                    padding: 10px;
                    text-align: left;
                    color: white;
                    font-size: 70%;
                }
                .custom-table th {
                    background-color: transparent;
                }
                .custom-table td {
                    background-color: transparent;
                }
                </style>
                """,
                unsafe_allow_html=True
            )

            # Creazione della tabella HTML con i dati
            table_html = "<table class='custom-table'>"
            for key, value in diagram.items():
                table_html += f"<tr><th>{key}</th><td>{value}</td></tr>"
            table_html += "</table>"

            # Mostrare la tabella in Streamlit
            st.markdown(f'<div style="width: 100%;">{table_html}</div>', unsafe_allow_html=True)

    st.write('Your previous annotations:')
    st.markdown(f"<h3 style='font-size: 0.8vw;'>{st.session_state.notes3}</h3>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='font-size: 0.8vw;'>{st.session_state.notes4}</h3>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='font-size: 0.8vw;'>{st.session_state.notes5}</h3>", unsafe_allow_html=True)

    if 'notes6' not in st.session_state:
        st.session_state.notes6 = "" 
    st.session_state.notes6 = st.text_area("Add notes. They'll be recorded.",
                                            value=st.session_state.notes6, height=100)
    
    selected_class = st.selectbox("Select a class:", [#'None',
                                                      'I will choose later',
                                                      'Grant',
                                                      'Deny'])
    
    cola, colb = st.columns([10, 1.5])
    with cola:  
        if st.button("Back"):
            if 'log10' not in st.session_state:
                st.session_state.log10 = log_interaction("temporary_notes_6", st.session_state.notes6) 
            if 'log11' not in st.session_state:
                st.session_state.log11 = log_interaction(f'Fourth impression: {selected_class}')
            log_interaction("Going back...")  
            go_to_page(5)
            st.rerun()  
    with colb:
        proceed_psix = st.button("Save")
    if proceed_psix:
        if selected_class != 'None':

            if 'log10' not in st.session_state:
                st.session_state.log10 = log_interaction("temporary_notes_6", st.session_state.notes6) 
            if 'log11' not in st.session_state:
                st.session_state.log11 = log_interaction(f'Fourth impression: {selected_class}')

            if selected_class != 'I will choose later':

                prepare_tuning(loaded=load_toEndlessTuning(idx=int(st.session_state.ID),
                                                            data_path=st.session_state.dbpath),
                                                            human_label=int(class_dict[selected_class]))

                go_to_page(10)
                st.rerun()
            else:
                go_to_page(10)
                st.rerun()
        else:
            st.markdown(f"<p style='font-size: 1vw;'>Error: select a class before proceeding!</p>", unsafe_allow_html=True)


####    END SESSION #####################################
def pagina_10():

    if 'notes6' in st.session_state:
        backpage = 6
    else:
        backpage=1


    if 'log10' in st.session_state:
        del st.session_state.log10
    if 'log11' in st.session_state:
        del st.session_state.log11

    st.title('Session ended')
    st.write('Click on "Download" to receive the csv file of your interactions. Double click on "New session" to begin a new session.')

    tuning_set = pd.read_csv('./dataset/Tuning_Set.csv')
    numero_file = len(pd.DataFrame(tuning_set))
    if numero_file >6:
        st.write('You have reached a sufficiently wide tuning set. Now, wait: the model is updating...')
        with st.spinner():
            DecisionTree_Tuning(data_path='./dataset/Tuning_Set.csv',origin='./dataset/Training_Data.csv',max_depth=4)
            st.success("Tuned successfully!")
            log_interaction('Model has been tuned.')
    else:   
        st.write(f'Up until now you have collected a set of {numero_file} items. When you will have collected more than 6 new items, a finetuning will automatically be carried out.')
   

    cola, colb, colc = st.columns([2,3.1,1])
    with cola:
        if st.button("Back"):
            log_interaction("Going back...")  
            go_to_page(backpage)  
            st.rerun()  
    with colb:

        try:
            with open('user_interactions.csv', mode='r', newline='') as file:
                f = file.read()  # Leggi il contenuto del file CSV

            # Mostra il pulsante per scaricare il file CSV
            st.download_button(
                label="Download", 
                data=f,  # Passa il contenuto del file come stringa
                file_name="your_interactions.csv", 
                mime="text/csv"
            )

        except FileNotFoundError:
            st.warning("No interaction log found. Are you sure it exists?")
    with colc:
        if st.button("Exit"):
                st.stop()
    
    if st.button("New session"):
        log_interaction('Beginning new session') 
        if 'exp' in st.session_state:
            del st.session_state.exp
        go_to_page(2)


              



 


options = ['None','Grant','Deny']
class_dict = {'Deny': '0', 
                       'Grant': '1'}
label_dict = {'0': 'Deny',
                       '1': 'Grant'}

if "pagina" not in st.session_state:
    print('Adding page session state...')
    st.session_state.pagina = 1 

# Logica per visualizzare la pagina corretta in base al valore di st.session_state.pagina
if st.session_state.pagina == 1:
    pagina_1()
elif st.session_state.pagina == 2: 
    pagina_2()
elif st.session_state.pagina == 3:
    pagina_3()
elif st.session_state.pagina == 4:
    pagina_4() 
elif st.session_state.pagina == 5:
    pagina_5()      
elif st.session_state.pagina == 6:
    pagina_6()
elif st.session_state.pagina == 10:
    pagina_10()
