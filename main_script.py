import argparse
from p_acquisition import m_acquisition
from p_analysis import m_analysis
from p_reporting import m_reporting

import os
import tensorflow as tf

import matplotlib.image as mpimg

import streamlit as st
import streamlit.components.v1 as components


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
            Made by <a class="white-text text-lighten-3" href="https://jcharistech.wordpress.com">Christian Javier Melo</a><br/>
            <a class="white-text text-lighten-3" href="https://jcharistech.wordpress.com">Christian Javier Melo</a>
            </div>
    </div>
    </footer>
    """


def main(args):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

    st.title("Vehicle type recognition in your environment")
    st.markdown("""This project consists in a **Streamlit visualization** that can be used
                to predict the vehicle type of your own image""")

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
        st.subheader("Vehicle type recognition through your pic")
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
                                             indexes=[20,22,29,31],
                                             class_names=label_names)
        directory_plots = sorted(os.listdir("/home/christian/Pictures/"))
        for each in directory_plots:
            st.image(mpimg.imread(f"/home/christian/Pictures/{each}"), width=None)
        print('done')

    elif choice == "About":
        st.subheader("About App")
        components.html(footer_temp, height=500)
        st.markdown("The following versions are to be defined. Possible paths to take are: "
                    "- Recognition of vehicle and road speed "
                    "- Recognition of vehicles and pedestrians")
        path = sorted(os.listdir("/home/christian/Downloads/next/"))
        for each in path:
            st.image(mpimg.imread(f"/home/christian/Downloads/next/{each}"), width=None)

    else:
        st.subheader("Home")
        html_temp = """
        <div style="background-color:royalblue;padding:20px;border-radius:20px">
        <h1 style="color:white;text-align:center;">IMVETY.PRO</h1>
        <h2 style="color:white;text-align:center;">Vehicle type recognition App</h2>
        </div>
        """

        components.html(html_temp)

        st.markdown("Recognition in any **environment**")
        components.html("""
            <html>
            <head>
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <style>
            * {box-sizing: border-box}
            body {font-family: Verdana, sans-serif; margin:0}
            .mySlides {display: none}
            img {vertical-align: middle}
            
            /* Slideshow container */
            .slideshow-container {
                max-width: 1000px;
                position: relative;
                margin: auto;
            }
            
            /* Next & previous buttons */
            .prev, .next {
                cursor: pointer;
                position: absolute;
                top: 50%;
                width: auto;
                padding: 16px;
                margin-top: -22px;
                color: white;
                font-weight: bold;
                font-size: 18px;
                transition: 0.6s ease;
                border-radius: 0 3px 3px 0;
                user-select: none;
            }
            
            /* Position the "next button" to the right */
            .next {
                right: 0;
                border-radius: 3px 0 0 3px;
            }
            
            /* On hover, add a black background color with a little bit see-through */
            .prev:hover, .next:hover {
                background-color: rgba(0,0,0,0.8);
            }
            
            /* Caption text */
            .text {
                color: #f2f2f2;
                font-size: 15px;
                padding: 8px 12px;
                position: absolute;
                bottom: 8px;
                width: 100%;
                text-align: center;
            }
            
            /* Number text (1/3 etc) */
            .numbertext {
                color: #f2f2f2;
                font-size: 12px;
                padding: 8px 12px;
                position: absolute;
                top: 0;
            }
            
            /* The dots/bullets/indicators */
            .dot {
                cursor: pointer;
                height: 15px;
                width: 15px;
                margin: 0 2px;
                background-color: #bbb;
                border-radius: 50%;
                display: inline-block;
                transition: background-color 0.6s ease;
            }
            
            .active, .dot:hover {
                background-color: #717171;
            }
            
            /* Fading animation */
            .fade {
                -webkit-animation-name: fade;
                -webkit-animation-duration: 1.5s;
                animation-name: fade;
                animation-duration: 1.5s;
            }
            
            @-webkit-keyframes fade {
                from {opacity: .4} 
                to {opacity: 1}
            }
            
            @keyframes fade {
                from {opacity: .4} 
                to {opacity: 1}
            }
            
            /* On smaller screens, decrease text size */
            @media only screen and (max-width: 300px) {
                .prev, .next,.text {font-size: 11px}
            }
            
            </style>
            </head>
            <body>
            
            <div class="slideshow-container">
            
            <div class="mySlides fade">
                <div class="numbertext">1 / 3</div>
                <img src="https://www.w3schools.com/howto/img_nature_wide.jpg" style="width:100%">
                <div class="text">Caption Text</div>
            </div>
            
            <div class="mySlides fade">
                <div class="numbertext">2 / 3</div>
                <img src="https://www.w3schools.com/howto/img_snow_wide.jpg" style="width:100%">
                <div class="text">Caption Two</div>
            </div>
            
            <div class="mySlides fade">
                <div class="numbertext">3 / 3</div>
                <img src="https://www.w3schools.com/howto/img_mountains_wide.jpg" style="width:100%">
                <div class="text">Caption Three</div>
            </div>
            
            <a class="prev" onclick="plusSlides(-1)">&#10094;</a>
            <a class="next" onclick="plusSlides(1)">&#10095;</a>
            
            </div>
            <br>
            
            <div style="text-align:center">
                <span class="dot" onclick="currentSlide(1)"></span> 
                <span class="dot" onclick="currentSlide(2)"></span> 
                <span class="dot" onclick="currentSlide(3)"></span> 
            </div>
            
            <script>
            var slideIndex = 1;
            showSlides(slideIndex);
            
            function plusSlides(n) {
                showSlides(slideIndex += n);
            }
            
            function currentSlide(n) {
                showSlides(slideIndex = n);
            }
            
            function showSlides(n) {
                var i;
                var slides = document.getElementsByClassName("mySlides");
                var dots = document.getElementsByClassName("dot");
                if (n > slides.length) {slideIndex = 1}    
                if (n < 1) {slideIndex = slides.length}
                for (i = 0; i < slides.length; i++) {
                    slides[i].style.display = "none";  
                }
                for (i = 0; i < dots.length; i++) {
                    dots[i].className = dots[i].className.replace(" active", "");
                }
                slides[slideIndex-1].style.display = "block";  
                dots[slideIndex-1].className += " active";
            }
            </script>
            </body>
            </html> 
            """)

        st.markdown("In this first version of the project, the objective has been to identify the main types of vehicles with a mobile device")
        path = sorted(os.listdir("/home/christian/Downloads/home/"))
        for each in path:
            st.image(mpimg.imread(f"/home/christian/Downloads/home/{each}"), width=None)


if __name__ == "__main__":
    arguments = argument_parser()
    main(arguments)
