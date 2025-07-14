import warnings
warnings.filterwarnings("ignore")#, category=DeprecationWarning) 

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.decomposition import PCA
import joblib
import matplotlib.pyplot as plt

"""NOTE: indexes in .csv files are always two steps ahead with 
respect to dataframes of the same .csv files opened here in Pandas."""

def create_dataset():

    data = pd.read_csv('./dataset/Loan_Dataset.csv')
    df = pd.DataFrame(data)
    df = df.dropna()
    df = df.drop_duplicates()
    df_zeros = df[df['Loan_Status'] == 0]
    df_surplus = df_zeros.sample(n=25000, random_state=42)
    df = df.drop(df_surplus.index)
    print(df['Loan_Status'].value_counts())

    labelencoders = {}

    for i in range(df.shape[1]-1):
        if type(df.iloc[0][i]) == str:
            labelencoder = LabelEncoder()
            labelencoder.fit(df[df.columns[i]])
            df[df.columns[i]] = labelencoder.transform(df[df.columns[i]])
            labelencoders[df.columns[i]]=labelencoder
        
    joblib.dump(labelencoders,'./models/dict_labelencoders.pkl')

    #TRAINING SET
    training_set = df.sample(n=int(len(df)*0.9), random_state=42)
    training_set.to_csv('./dataset/Training_Data.csv',index=False)   
    print('Training set created!')

    #CASE STUDIES
    remainder = df.drop(training_set.index)
    sampled_0 = remainder[remainder[remainder.columns[-1]] == 0].sample(n=100, random_state=42)
    sampled_1 = remainder[remainder[remainder.columns[-1]] == 1].sample(n=100, random_state=42)
    case_studies = pd.concat([sampled_0, sampled_1])
    case_studies.to_csv('./dataset/Case_Studies.csv',index=False)
    print('Case Studies ready!')

    #TEMPORARY SET
    temporary_set = remainder.drop(case_studies.index)
    temporary_set.to_csv('./dataset/Temporary_Set.csv',index=False)
    print('Temporary set created!')
    
    #TUNING SET
    tuning_set = pd.DataFrame(columns=df.columns)
    tuning_set.to_csv('./dataset/Tuning_Set.csv', index=False)
    return None



def load_data(data_path):
    
    #create_dataset()
    data = pd.read_csv(data_path)
    data = pd.DataFrame(data)
    X = pd.DataFrame(data.iloc[:,:-1])
    y = pd.Series(data.iloc[0:,-1])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test,X,y



def pretrain_model(X_train, X_test, y_train, y_test, max_depth):
    
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model trained with test accuracy: {accuracy:.4f}")
    joblib.dump(model,'./models/DecisionTree.pkl')



def get_explanation(idx,data_path):
    
    """tree_.feature è l'indice della feature che ha determinato la divisione.
        Se l'indice è -2, convenzionalmente è un nodo foglia, non ci sono features
        né condizioni su quel nodo. tree_.threshold è invece il valore di soglia
        della condizione: l'indice di ogni item nella lista di threshold è
        l'indice del nodo presso cui questa soglia è implementata. Se -2,
        non c'è soglia o non ha valore."""

    model = joblib.load('./models/DecisionTree.pkl')
    loaded_labelencoders = joblib.load('./models/dict_labelencoders.pkl')
    inst = load_toEndlessTuning(idx,data_path)
    node_indices = model.decision_path(inst).indices   
    feature_index = []
    thresholds = []
    for node in node_indices:
        feature_index.append(model.tree_.feature[node] )
        thresholds.append(model.tree_.threshold[node])

    if -2 in feature_index:
        leaf = feature_index.index(-2)
        thresholds.remove(thresholds[leaf])
        node_indices = np.delete(node_indices,leaf)
        feature_index.remove(feature_index[leaf])

    feature_names = [inst.columns[i] for i in feature_index]
    feature_values = [inst[i].values[0] for i in feature_names]
    decoded_values = feature_values.copy()
    for i in feature_names:
        if i in loaded_labelencoders: 
            decoded_values[feature_names.index(i)] = loaded_labelencoders[i].inverse_transform(inst[i].values)[0]

    """Deduzione della condizione. Uno split è un'implicita asserzione di verità. 
    Adesso si confronteranno i feature values con le soglie. Se il valore sarà
    <=, tale sarà intesa la condizione. Lo stesso vale per il >. Anzi, rispetto al
    significato dello split non pare scorretto neanche distinguere < e =, in quanto
    per l'appunto l'asserzione è vera. Notare che i dati
    non sono stati standardizzati per addestrare l'albero decisionale, quindi i dati
    originariamente numerici possono essere confrontati col valore di soglia 
    senza difficoltà."""

    split_conditions = []
    for i in range(len(feature_values)):
        if feature_values[i] < thresholds[i]:
            if type(decoded_values[i]) == str:
                split_conditions.append(':')
            else:
                split_conditions.append(' is less than')
        elif feature_values[i] == thresholds[i]:
            if type(decoded_values[i]) == str:
                split_conditions.append(':')
            else:
                split_conditions.append(' is equal to')
        else:
            if type(decoded_values[i]) == str:
                split_conditions.append(':')
            else:
                split_conditions.append(' is greater than')

    #print('Node: ',node_indices)
    #print('Feature: ',feature_names)
    #print('Encoded values: ', feature_values)
    #print('Original values: ', decoded_values)
    #print('Threshold: ',thresholds)

    rule = []
    for i in range(len(feature_names)):
        if type(decoded_values[i]) == str:
            rule.append(feature_names[i]+split_conditions[i]+' '+decoded_values[i])
        else:
            rule.append(feature_names[i]+split_conditions[i]+' '+str(round(thresholds[i],2)))

    return rule


def get_confidence(idx,data_path):
    
    model = joblib.load('./models/DecisionTree.pkl')
    loaded = load_toEndlessTuning(idx=idx,data_path=data_path)
    confidence = model.predict_proba(loaded)
    classes = model.classes_
    return confidence, classes



def get_embeddings(X_train):
    
    pca = PCA(n_components=2)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_scaled = scaler.transform(X_train)
    X_pca = pca.fit(X_scaled)
    embeddings = X_pca.transform(X_scaled)
    joblib.dump(embeddings,'./models/embeddings.pkl')
    joblib.dump(X_pca,'./models/pca.pkl')
    joblib.dump(scaler,'./models/scaler.pkl')
    print('Got embeddings!')



def similarity_checker(idx,data_path,origin,num_points):

    loaded_pca = joblib.load('./models/pca.pkl')
    loaded_embeddings = joblib.load('./models/embeddings.pkl')
    loaded_scaler = joblib.load('./models/scaler.pkl')         
    loaded_labelencoders = joblib.load('./models/dict_labelencoders.pkl')
    case_studies = pd.read_csv(data_path)
    case_studies = pd.DataFrame(case_studies)
    origin = pd.read_csv(origin)
    origin = pd.DataFrame(origin)
    #print(origin[:3])
    to_embed = case_studies.iloc[idx,:-1].values
    to_embed = to_embed.reshape(1, -1)
    to_embed = loaded_scaler.transform(to_embed)
    print(f'Example of loaded embeddings: {loaded_embeddings[0]}')
    embedded = loaded_pca.transform(to_embed)
    print(f'Embedding of the current case study: {embedded}')

    #distanze dal case study
    distances = [euclidean_distances(i.reshape(1,-1),embedded) for i in loaded_embeddings]
    smallest_indices = [index for index, value in sorted(enumerate(distances), key=lambda x: x[1])[:num_points]] 
    zipped_distances = list(zip(loaded_embeddings,distances))

    x_values = []
    y_values = []
    cs_x = embedded[0][0]
    cs_y = embedded[0][1]
    similar_instances = pd.DataFrame()

    for i in smallest_indices:
        x_values.append(zipped_distances[i][0][0])
        y_values.append(zipped_distances[i][0][1])
        similar_instances = similar_instances._append(origin.iloc[[i]])
    
    for j in loaded_labelencoders:
        similar_instances[j] = loaded_labelencoders[j].inverse_transform(similar_instances[j])

    return x_values,y_values,cs_x,cs_y,similar_instances



def inverse_relabeler(idx,data_path):

    """Ricorda che le etichette del dato esaminato non devono essere ulteriori rispetto a 
    quelle dei dati su cui si è fittato il label_encoder di scikit-learn."""

    loaded_labelencoders = joblib.load('./models/dict_labelencoders.pkl')
    applicant = load_toEndlessTuning(idx,data_path)
    for j in loaded_labelencoders:
        applicant[j] = loaded_labelencoders[j].inverse_transform(applicant[j])

    return applicant



def load_toEndlessTuning(idx,data_path):

    df = pd.read_csv(data_path)
    df = pd.DataFrame(df)
    loaded = df.iloc[[idx]]
    loaded = loaded.drop(loaded.columns[-1],axis=1)

    return loaded



def prepare_tuning(loaded,human_label):

    """Il codice è meramente sperimentale e non è fatto 
    per tornare sulla stessa immagine prima del fine-tuning. 
    La decisione dell'utente, dunque, nei termini 
    dell'esperimento, dev'essere definitiva. Per 
    un'implementazione più seria dovrà essere resa provvisoria.
    
    Trattandosi di Random Forest, è difficile parlare di un
    vero e proprio fine-tuning. Piuttosto, si tratterà di un rehearsal. 
    O meglio, dipenderà dai dati inviati al modello: si potrà decidere
    di inviare anche dati parzialmente già visti, oppure solo
    dati nuovi. In questo caso, si sceglie di inviare solo dati mai
    osservati dal Random Forest Classifier, estraendo dal Temporary_Set.csv,
    volta per volta, una riga etichettata in maniera complementare (il task
    è infatti binary) a quella predetta dall'esperto sul caso di studio."""

    #Deve essere un'istanza DataFrame senza etichetta, dunque senza l'ultima colonna
    #print(type(loaded))
    loaded['Loan_Status']=human_label
    tuning_set = pd.read_csv('./dataset/Tuning_Set.csv').drop(columns=["Unnamed: 0"], errors="ignore")
    tuning_set = pd.DataFrame(tuning_set).drop(columns=["Unnamed: 0"], errors="ignore")
    temporary_ = pd.read_csv('./dataset/Temporary_Set.csv')
    temporary_ = pd.DataFrame(temporary_)

    tuning_set = pd.concat([tuning_set,loaded],ignore_index = False)

    if human_label == 0:
        other_class = temporary_[temporary_['Loan_Status']==1]
        other_class = other_class.sample(n=1,random_state=42)
        temporary_ = temporary_.drop(other_class.index)
    elif human_label == 1:
        other_class = temporary_[temporary_['Loan_Status']==0]
        other_class = other_class.sample(n=1,random_state=42)
        temporary_ = temporary_.drop(other_class.index)
    else:
        print('Error selecting label!')
    tuning_set = pd.concat([tuning_set,other_class],ignore_index=False)
    tuning_set.to_csv('./dataset/Tuning_Set.csv')
    temporary_.to_csv('./dataset/Temporary_Set.csv',index=False)



def DecisionTree_Tuning(data_path,origin,max_depth):

    """In questo caso, a differenza che nelle reti neurali,
    l'appendimento è rigido e la flessibilità va cercata nei
    dati. Di conseguenza, il modello nuovo va addestrato su
    tutti i dati nuovi ed una piccola porzione, qui 20%,
    di dati vecchi."""

    tune_data = pd.read_csv(data_path)
    tune_data = pd.DataFrame(tune_data)

    origin = pd.read_csv(origin)
    origin = pd.DataFrame(origin)
    X_add0 = origin[origin['Loan_Status']==0].sample(len(tune_data)//10)
    X_add1 = origin[origin['Loan_Status']==1].sample(len(tune_data)//10)
    tune_data = pd.concat([tune_data,X_add0,X_add1],ignore_index=False)
    tune_data = tune_data.drop(columns=["Unnamed: 0"], errors="ignore")
    X = pd.DataFrame(tune_data.iloc[:,:-1])
    y = pd.Series(tune_data.iloc[0:,-1])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = DecisionTreeClassifier(max_depth=max_depth,random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    #confidences = model.predict_proba(X_test) 
    print(f"Model tuned with test accuracy: {accuracy:.4f}")
    joblib.dump(model,'./models/DecisionTree_Tuned.pkl')



if __name__ == "__main__":

    create_dataset()
    X_train, X_test, y_train, y_test,X,y = load_data(data_path='./dataset/Training_Data.csv')
    pretrain_model(X_train, X_test, y_train, y_test,max_depth=4)
    get_explanation(5,data_path='./dataset/Case_Studies.csv')
    get_embeddings(X_train=X_train)
    similarity_checker(idx=4,data_path='./dataset/Case_Studies.csv',origin='./dataset/Training_Data.csv',num_points=3)
    prepare_tuning(loaded=load_toEndlessTuning(65,data_path='./dataset/Case_Studies.csv'),human_label=1)
    DecisionTree_Tuning(data_path='./dataset/Tuning_Set.csv',origin='./dataset/Training_Data.csv',max_depth=4)
    pass