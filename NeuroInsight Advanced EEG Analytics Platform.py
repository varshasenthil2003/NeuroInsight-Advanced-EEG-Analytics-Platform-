import streamlit as st
import pandas as pd
from scipy.stats import spearmanr, kendalltau
import streamlit as st
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from scipy.linalg import eigh
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.cluster import KMeans


st.set_page_config(
    page_title="Custom Streamlit Styling",
    page_icon=":smiley:",
    layout="centered",
    initial_sidebar_state="expanded",
)
st.set_option('deprecation.showPyplotGlobalUse', False)

def home():

        #Comment this after running one time
        def normalize_dataframe(df):
            numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
            for column in numerical_columns:
                mean = df[column].mean()
                df[column].fillna(mean, inplace=True)

                max_allowed_value = 1e10  
                df[column] = np.clip(df[column], -max_allowed_value, max_allowed_value)
                
                mean = df[column].mean()
                std = df[column].std()
                
                df[column] = (df[column] - mean) / std


        def normalize_csv(input_csv_path):
            df = pd.read_csv(input_csv_path)
            normalize_dataframe(df)
            df.to_csv(input_csv_path, index=False) 


        folder1_path = 'C:\\Users\\birth\\ML\\ADHD'
        folder2_path = 'C:\\Users\\birth\\ML\\Control'


        for folder_path in [folder1_path, folder2_path]:
            for filename in os.listdir(folder_path):
                if filename.endswith(".csv"):
                    input_csv_path = os.path.join(folder_path, filename)
                    normalize_csv(input_csv_path)
        #Till this comment it
        st.title("EEG Data Analysis and Modelling")
        st.write("By 21PD22 - Nilavini")
        st.write("   21PD27 - Raja Neha")
        st.write("   21PD39 - Varsha")
        st.header("About the dataset")

        with open("style.css", "r") as f:
            css = f.read()
            st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

        st.markdown("""<div class="boxed-paragraph">Participants were 61 children with ADHD and 60 healthy controls (boys and girls, ages 7-12). EEG recording was performed based on 10-20 standard by 19 channels (Fz, Cz, Pz, C3, T3, C4, T4, Fp1, Fp2, F3, F4, F7, F8, P3, P4, T5, T6, O1, O2) at 128 Hz sampling frequency. The A1 and A2 electrodes were the references located on earlobes.Since one of the deficits in ADHD children is visual attention, the EEG recording protocol was based on visual attention tasks.    In the task, a set of pictures of cartoon characters was shown to the children and they were asked to count the characters. The number of characters in each image was randomly selected between 5 and 16, and the size of the pictures was large enough to be easily visible and countable by children. To have a continuous stimulus during the signal recording, each image was displayed immediately and uninterrupted after the child's response.  Thus, the duration of EEG recording throughout this cognitive visual task was dependent on the child's performance (i.e. response speed)""", unsafe_allow_html=True)
        
        option = st.radio('Select a Class Type :',
                  ['ADHD','Control'])
        if option == 'ADHD':
            
            dataset = pd.read_csv('C:\\Users\\birth\\ML\\ADHD\\v1p.csv')
            st.write("\nFirst 5 rows of the dataset for ADHD:")
            st.write(dataset.head(5))

        elif option == 'Control':
            
            dataset = pd.read_csv('C:\\Users\\birth\\ML\\Control\\v41p.csv')
            st.write("\nFirst 5 rows of the dataset for Control:")
            st.write(dataset.head(5))
            
        
        
            

def NonlinearCorrCoeff():
    
    #Function to find the minimum number of rows among CSV files in a folder
    def find_min_rows_in_folder(folder_path):
        min_rows = float('inf')  
        min_csv_file = None

        for filename in os.listdir(folder_path):
            if filename.endswith(".csv"):
                file_path = os.path.join(folder_path, filename)
                df = pd.read_csv(file_path)
                num_rows = len(df)
                if num_rows < min_rows:
                    min_rows = num_rows
                    min_csv_file = file_path

        return min_csv_file, min_rows

    # Function to normalize a CSV file to a specific number of rows
    def normalize_csv(input_csv_path, target_num_rows):
        df = pd.read_csv(input_csv_path)
        df = df.iloc[:, 1:]   # Comment this after one time running it 
        normalized_df = df.sample(target_num_rows, replace=True)
        normalized_df.to_csv(input_csv_path, index=False)  # Overwrite the original file

    # Paths to your folders with 60 CSV files each
    folder1_path = 'C:\\Users\\birth\\ML\\ADHD'
    folder2_path = 'C:\\Users\\birth\\ML\\Control'

    # Find the CSV file with the minimum number of rows
    min_csv_file1, min_rows1 = find_min_rows_in_folder(folder1_path)
    min_csv_file2, min_rows2 = find_min_rows_in_folder(folder2_path)

    # Determine the overall minimum number of rows among all CSV files
    if min_rows1 < min_rows2:
        min_csv_file = min_csv_file1
        min_rows = min_rows1
    else:
        min_csv_file = min_csv_file2
        min_rows = min_rows2

    # Normalize all CSV files to the minimum number of rows
    for folder_path in [folder1_path, folder2_path]:
        for filename in os.listdir(folder_path):
            if filename.endswith(".csv"):
                input_csv_path = os.path.join(folder_path, filename)
                normalize_csv(input_csv_path, min_rows)

    print(f"Minimum number of rows found: {min_rows}")
    print(f"CSV file with minimum rows: {min_csv_file}")
    


    # Function to calculate the Spearman and Kendall correlation matrices for a CSV file
    def calculate_nonlinear_correlations(input_csv_path):
        df = pd.read_csv(input_csv_path)
        spearman_corr = df.corr(method='spearman')
        kendall_corr = df.corr(method='kendall')
        return spearman_corr, kendall_corr




    def calculate_nonlinear_correlations_in_folder(folder_path, min_rows):
        spearman_correlations = {}
        kendall_correlations = {}
        count_spearman = 0
        count_kendall = 0
        
        for filename in os.listdir(folder_path):
            if filename.endswith(".csv"):
                input_csv_path = os.path.join(folder_path, filename)
                spearman_corr, kendall_corr = calculate_nonlinear_correlations(input_csv_path)
                spearman_correlations[filename] = spearman_corr
                kendall_correlations[filename] = kendall_corr
                count_spearman += 1
                count_kendall += 1

        return spearman_correlations, kendall_correlations, count_spearman, count_kendall


    spearman_correlations_folder1, kendall_correlations_folder1, count_spearman1, count_kendall1 = calculate_nonlinear_correlations_in_folder(folder1_path, min_rows)
    spearman_correlations_folder2, kendall_correlations_folder2, count_spearman2, count_kendall2 = calculate_nonlinear_correlations_in_folder(folder2_path, min_rows)

                
            
    def plot_heatmap(correlation_matrix, title):
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', cbar=True)
        plt.title(title)
        plt.show()
        st.pyplot()

# Function to select a folder dynamically
    def select_folder():
        folder = st.selectbox("Select a folder:", ["Folder 1", "Folder 2"])
        return folder

    # Function to select a CSV file dynamically from the specified folder
    def select_csv_file(folder):
        folder_name = "ADHD" if folder == 1 else "Control"

        # Specify the full path to the folders
        folder_path = "C:\\Users\\birth\\ML"
        # List all CSV files in the folder
        csv_files = [file for file in os.listdir(os.path.join(folder_path, folder_name)) if file.endswith(".csv")]

        if not csv_files:
            print(f"No CSV files found in {folder_name}.")
            return None

        print(f"Select a CSV file from {folder_name}:")
        for i, file in enumerate(csv_files):
            print(f"{i + 1}: {file}")

        selected_filename = st.selectbox(f"Select a CSV file from {folder_name}:", csv_files)
        return selected_filename

    # Function to plot Spearman and Kendall correlation heatmaps for a selected CSV file in a folder
    def plot_correlation_heatmaps(folder, selected_filename):
        folder_name = "Folder1" if folder == 1 else "Folder2"

        # Spearman Correlation
        spearman_correlation_matrix = spearman_correlations_folder1.get(selected_filename) if folder == 1 else spearman_correlations_folder2.get(selected_filename)
        if spearman_correlation_matrix is not None:
            spearman_title = f"Spearman Correlation Heatmap {'ADHD' if folder == 1 else 'Control'} - {selected_filename}"
            plot_heatmap(spearman_correlation_matrix, spearman_title)

        # Kendall Correlation
        kendall_correlation_matrix = kendall_correlations_folder1.get(selected_filename) if folder == 1 else kendall_correlations_folder2.get(selected_filename)
        if kendall_correlation_matrix is not None:
            kendall_title = f"Kendall Correlation Heatmap {'ADHD' if folder == 1 else 'Control'} - {selected_filename}"
            plot_heatmap(kendall_correlation_matrix, kendall_title)

    # Example usage:
    folder = select_folder()  # Dynamically select the folder
    if folder is not None:
        selected_filename = select_csv_file(folder)  # Dynamically select the CSV file
        if selected_filename is not None:
            plot_correlation_heatmaps(folder, selected_filename)



def Topoplot():
    st.title("Topo plot")
    st.write('Select:')
    option_1 = st.checkbox('ADHD')
    option_2 = st.checkbox('Control')
    
    image1 = Image.open('topoplotadhd.png')
    image2 = Image.open('topopltControl.png')
    
    st.write("The color intensity or contour lines in the topo plots can reveal the spatial distribution of electrical activity on the scalp. Darker colors or closely spaced contour lines may indicate regions of high activity, while lighter colors or widely spaced lines may indicate low activity.")
    
    if option_1 and option_2:
        # Display both images side by side
        st.write("Comparison between ADHD and Control Topoplot for easy visualization")
        st.write("So we can observe by comparing both the Topo plots that children with ADHD higher brain activity")
        col1, col2 = st.columns(2)
        with col1:
            st.image(image1, caption='ADHD Topoplot', use_column_width=True)
        with col2:
            st.image(image2, caption='Control Topoplot', use_column_width=True)
    elif option_1:
        # Display only the ADHD image
        st.image(image1, caption='ADHD Topoplot', use_column_width=True)
    elif option_2:
        # Display only the Control image
        st.image(image2, caption='Control Topoplot', use_column_width=True)

    
    
def Spectral_Clustering():
    st.title("Spectral Clustering")
    
    directory_path = "C:\\Users\\birth\\ML"

    median_values_adhd = []
    median_values_control = []

    for dir_name in os.listdir(directory_path):
        dir_path = os.path.join(directory_path, dir_name) 
        if os.path.isdir(dir_path):
            is_adhd_group = "ADHD" in dir_name

                    
            for filename in os.listdir(dir_path):
                if filename.endswith(".csv"):
                    file_path = os.path.join(dir_path, filename)
                    try:
                        df = pd.read_csv(file_path)
                        df = df.iloc[:, :19]

                                
                        if is_adhd_group:
                            median_values_adhd.append(df.median(axis=0))
                        else:
                            median_values_control.append(df.median(axis=0))

                                
                        print(f"Median values (excluding columns 19 and beyond) for {filename} in group {dir_name}:\n{df.median(axis=0)}")
                    except FileNotFoundError:
                        print(f"File {filename} not found.")
                    except Exception as e:
                        print(f"An error occurred for {filename} in group {dir_name}: {str(e)}")

    adhd_matrix = np.array(median_values_adhd)
    control_matrix = np.array(median_values_control)
    
    

    print(adhd_matrix)
    print(control_matrix)

    adhd_shape = adhd_matrix.shape
    control_shape = control_matrix.shape

    print("Dimensions of ADHD Matrix:", adhd_shape)
    print("Dimensions of Control Matrix:", control_shape)

            

    adhd_matrix = np.array(median_values_adhd)
    control_matrix = np.array(median_values_control)
    
   

    bandwidth = 1.0  


    pairwise_distances_adhd = euclidean_distances(adhd_matrix)
    similarity_matrix_adhd = np.exp(- (pairwise_distances_adhd * 2) / (2 * bandwidth * 2))

    pairwise_distances_control = euclidean_distances(control_matrix)
    similarity_matrix_control = np.exp(- (pairwise_distances_control * 2) / (2 * bandwidth * 2))


    def laplacian_matrix(similarity_matrix):
        # Calculate the degree matrix
        degree_matrix = np.diag(np.sum(similarity_matrix, axis=1))
        
        # Calculate the Laplacian matrix
        laplacian = degree_matrix - similarity_matrix
        
        return laplacian

    laplacian_adhd = laplacian_matrix(similarity_matrix_adhd)
    laplacian_control = laplacian_matrix(similarity_matrix_control)


    print(laplacian_adhd)


    laplacian_adhd = laplacian_matrix(similarity_matrix_adhd)
    laplacian_control = laplacian_matrix(similarity_matrix_control)

    num_clusters = 2

    eigenvalues_adhd, eigenvectors_adhd = eigh(laplacian_adhd)
    eigenvalues_control, eigenvectors_control = eigh(laplacian_control)

    smallest_eigenvalues_adhd = eigenvalues_adhd[:num_clusters]
    smallest_eigenvectors_adhd = eigenvectors_adhd[:, :num_clusters]

    smallest_eigenvalues_control = eigenvalues_control[:num_clusters]
    smallest_eigenvectors_control = eigenvectors_control[:, :num_clusters]


    embedding_adhd = PCA(n_components=num_clusters).fit_transform(smallest_eigenvectors_adhd)
    embedding_control = PCA(n_components=num_clusters).fit_transform(smallest_eigenvectors_control)


    cluster_labels_adhd = SpectralClustering(n_clusters=num_clusters, affinity='nearest_neighbors').fit_predict(embedding_adhd)
    cluster_labels_control = SpectralClustering(n_clusters=num_clusters, affinity='nearest_neighbors').fit_predict(embedding_control)


    # Scatter plot for ADHD data
    plt.figure(figsize=(8, 6))
    plt.scatter(embedding_adhd[:, 0], embedding_adhd[:, 1], c=cluster_labels_adhd, cmap='viridis')
    plt.title('Spectral Clustering Results for ADHD')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.colorbar()
    plt.show()
    st.pyplot()

    # Scatter plot for control data
    plt.figure(figsize=(8, 6))
    plt.scatter(embedding_control[:, 0], embedding_control[:, 1], c=cluster_labels_control, cmap='viridis')
    plt.title('Spectral Clustering Results for Control')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.colorbar()
    plt.show()
    st.pyplot()


def SVM_and_rank():
    st.title("Rank")
    directory_path = "C:\\Users\\birth\\ML"

    median_values_adhd = []
    median_values_control = []

    for dir_name in os.listdir(directory_path):
        dir_path = os.path.join(directory_path, dir_name) 
        if os.path.isdir(dir_path):
            is_adhd_group = "ADHD" in dir_name

                    
            for filename in os.listdir(dir_path):
                if filename.endswith(".csv"):
                    file_path = os.path.join(dir_path, filename)
                    try:
                        df = pd.read_csv(file_path)
                        df = df.iloc[:, :19]

                                
                        if is_adhd_group:
                            median_values_adhd.append(df.median(axis=0))
                        else:
                            median_values_control.append(df.median(axis=0))

                                
                        print(f"Median values (excluding columns 19 and beyond) for {filename} in group {dir_name}:\n{df.median(axis=0)}")
                    except FileNotFoundError:
                        print(f"File {filename} not found.")
                    except Exception as e:
                        print(f"An error occurred for {filename} in group {dir_name}: {str(e)}")

    adhd_matrix = np.array(median_values_adhd)
    control_matrix = np.array(median_values_control)
    
    

    print(adhd_matrix)
    print(control_matrix)

    adhd_shape = adhd_matrix.shape
    control_shape = control_matrix.shape

    print("Dimensions of ADHD Matrix:", adhd_shape)
    print("Dimensions of Control Matrix:", control_shape)

            

    adhd_matrix = np.array(median_values_adhd)
    control_matrix = np.array(median_values_control)
    adhd_rank = np.linalg.matrix_rank(adhd_matrix)
    control_rank = np.linalg.matrix_rank(control_matrix)
    # Print the rank
    st.write(f"The rank of the median matrix of adhd is: {adhd_rank}")
    st.write(f"The rank of the median matrix of control is: {control_rank}")
    
    st.write("Result of calculating the rank of 121x19 median matrix is 19, it means that all 19 columns in the matrix are linearly independent. In other words, there are no linear dependencies among these columns.Having a rank equal to the number of columns (19 in this case) implies that each column provides unique and essential information")

    st.title("SVM")
    
    labels_adhd = [0] * len(adhd_matrix)
    labels_control = [1] * len(control_matrix)

    # Combine the labels into a single target array 'y'
    y = np.concatenate((labels_adhd, labels_control))
    print(y)
    # Perform feature selection using SelectKBest
    num_features_to_select = 10
    selector = SelectKBest(f_classif, k=num_features_to_select)
    median_matrix_selected = selector.fit_transform(np.concatenate((adhd_matrix, control_matrix)), y)


    print(median_matrix_selected)
    
    X_train, X_test, y_train, y_test = train_test_split(median_matrix_selected, y, test_size=0.2, random_state=42)

    svm_classifier = SVC(kernel='linear', C=1.0)  # You can choose different kernel functions and C values based on your data and problem.

    svm_classifier.fit(X_train, y_train)
    y_pred = svm_classifier.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)

    st.write("Accuracy:", accuracy)
    st.text("Classification Report:\n" + classification_rep)
        
def main():

        st.sidebar.title("Analysis")
        page = st.sidebar.selectbox("Select a Page", ["About the Dataset", "Nonlinear Correlation Coefficient ", "Topo plot","Spectral Clustering","SVM and rank"])
            
        if page == "About the Dataset":
                home()
        elif page == "Nonlinear Correlation Coefficient ":
                NonlinearCorrCoeff()
        elif page == "Topo plot":
                Topoplot()
        elif page == "Spectral Clustering":
                Spectral_Clustering()
        elif page == "SVM and rank":
                SVM_and_rank()
        

if __name__ == "__main__":
    
    main()      