import os
import matplotlib.image as mpimg
import tensorflow as tf
import streamlit as st
import streamlit.components.v1 as components
import argparse
from p_acquisition import m_acquisition
from p_analysis import m_analysis
from p_reporting import m_reporting


def argument_parser():
    # Parameters
    parser = argparse.ArgumentParser(description='Vehicle Type Recognition')
    parser.add_argument('--num_classes', default='15', type=int, help='Number of classes')
    parser.add_argument('--model', default='model', type=str, help='Model name')
    parser.add_argument('--train', default='./data/raw/vehicle/train/train/', type=str, help='Directory of train data')
    parser.add_argument('--test', default='./data/raw/vehicle/test/proof/', type=str, help='Directory of test data')
    parser.add_argument('--IMG_HEIGHT', default='224', type=int, help='Image height')
    parser.add_argument('--IMG_WIDTH', default='224', type=int, help='Image width')
    parser.add_argument('--batch_size', default='32', type=int, help='Batch size')
    parser.add_argument('--epochs', default='100', type=int, help='Epochs')
    parser.add_argument('--weight_decay', default='1e-4', type=float, help='Weight decay')

    args = parser.parse_args()
    return args


footer_temp = """
    <link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/materialize/1.0.0/css/materialize.min.css" type="text/css" rel="stylesheet" media="screen,projection"/>
    <link href="static/css/style.css" type="text/css" rel="stylesheet" media="screen,projection"/>
        <link rel="stylesheet" href="https://use.fontawesome.com/releases/v5.5.0/css/all.css" integrity="sha384-B4dIYHKNBt8Bc12p+WXckhzcICo0wtJAoU8YZTY5qE0Id1GSseTk6S+L3BlXeVIU" crossorigin="anonymous">
    <footer class="page-footer grey darken-4">
        <div class="container" id="aboutapp">
            <div class="row">
                <div class="col l6 s12">
                    <h5 class="white-text">About Vehicle Type Recognition App</h5>
                    <p class="grey-text text-lighten-4">Using Streamlit, Keras and Tensorflow.</p>
        </div>

    <div class="col l3 s12">
                    <h5 class="white-text">Connect With Me</h5>
                    <ul>
                        <a href="https://github.com/ChristianJavierMelo" target="_blank" class="white-text">
                        <i class="fab fa-github-square fa-4x"></i>
                    </a>
                    <a href="https://www.linkedin.com/in/christian-javier-melo" target="_blank" class="white-text">
                        <i class="fab fa-linkedin fa-4x"></i>
                    </a>
                    <a href="https://twitter.com/ChristianJM12" target="_blank" class="white-text">
                        <i class="fab fa-twitter-square fa-4x"></i>
                    </a>
                    <a href="https://facebook.com/christianjaviermelo" target="_blank" class="white-text">
                        <i class="fab fa-facebook fa-4x"></i>
                    </a>
                    </ul>
                    </div>
                </div>
            </div>
    <div class="footer-copyright">
        <div class="container">
            Made by <a class="white-text text-lighten-3" href="https://www.linkedin.com/in/christian-javier-melo/">Christian Javier Melo</a><br/>
            <a class="white-text text-lighten-3" href="https://www.linkedin.com/in/christian-javier-melo/">Christian Javier Melo</a>
            </div>
    </div>
    </footer>
    """


def main(args):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

    st.title("Vehicle Type Recognition")
    st.markdown(""">Project based on **Machine Learning algorithms** shown through **Streamlit** library.""")

    menu = ["Home", "Data information", "Recognition App", "About"]
    choice = st.sidebar.selectbox("Menu", menu)
    if choice == "Data information":
        st.subheader("Data Information of vehicle dataset")
        print('starting pipeline...')
        # Load image generators
        print('loading images...')
        train_dataset = m_acquisition.VehicleDataset(args)
        print('loaded successfully')

        # Train model
        print('training model')
        train_model = m_analysis.train(args, train_dataset.train_generator, train_dataset.validation_generator)
        print('model trained successfully')

        # Plot the features of the model
        # chart_features_model = m_analysis.plot_accuracy(args, train_model)

    elif choice == "Recognition App":
        st.markdown("""Vehicle type recognition through a photo made by us or whatever photo 
        where the vehicle appears in the foreground.""")
        # Predict values
        print("preparing the data to predict...")
        test_dataset = m_reporting.test_dataframe(args)
        print("...working on predict...")
        proof = m_reporting.predicted_data(args, test_dataset)
        print('...predictions successfully')
        label_data = m_reporting.predicted_label(test_dataset, proof)
        label_names = ['Ambulance', 'Bicycle', 'Boat', 'Bus', 'Car', 'Helicopter', 'Limousine',
                       'Motorcycle', 'PickUp', 'Segway', 'Snowmobile', 'Tank', 'Taxi', 'Truck', 'Van']
        # Ploting values
        print("plot the result...")
        result = m_reporting.plot_image_pred(args,
                                             label_data,
                                             proof,
                                             indexes=[31],
                                             class_names=label_names)
        directory_plots = sorted(os.listdir("/home/christian/Pictures/"))
        for each in directory_plots:
            st.image(mpimg.imread(f"/home/christian/Pictures/{each}"), width=None)
        print('done')

    elif choice == "About":
        st.subheader("About Project")
        components.html(footer_temp, height=500)
        st.markdown("The following versions are to be defined. Possible paths to take are:")
        path = sorted(os.listdir("/home/christian/Downloads/next/"))
        st.markdown("- Road speed according to each predicted vehicle")
        for each in path:
            st.image(mpimg.imread(f"/home/christian/Downloads/next/{each}"), width=None)
        st.markdown("- Vehicles, pedestrians and sign lights recognition")

    else:
        st.subheader("Home")
        html_temp = """
        <div style="background-color:royalblue;padding:10px;border-radius:10px">
        <h1 style="color:white;text-align:center;">Transport vehicle prediction</h1>
        <h2 style="color:white;text-align:center;">What kind of vehicle is it?</h2>
        </div>
        """

        components.html(html_temp)

        st.markdown("""In this *first version* of the project, **the objective** consists of 
        **predicting the type of transport vehicle from an image** (between 15 categories) in order to identify 
        the vehicle in any environment""")
        path = sorted(os.listdir("/home/christian/Downloads/home/"))
        for each in path:
            st.image(mpimg.imread(f"/home/christian/Downloads/home/{each}"), width=None)


if __name__ == "__main__":
    arguments = argument_parser()
    main(arguments)
