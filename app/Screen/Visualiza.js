import React, { Component } from 'react';
import {
        View,
        StyleSheet,
        Image 
    } from 'react-native';

import { URL_API } from '../Utils/url_api';

// Http request
const urlGetImagem = `${URL_API}/imagem/baixar`;

class Visualiza extends Component {

    static navigationOptions = {

        title: 'Visualiza',
        headerStyle: {
          backgroundColor: '#39b500',
        },
        headerTintColor: 'white',
        headerTitleStyle: {
          fontWeight: 'bold',
          fontSize: 30
        },

    };

    state = {

        imgPath: '',

    };

    /**
     * Método para pegar os dados vindo de outra tela e processar os dados
     * @author Pedro Biasutti
     */
    async componentDidMount() {

        const { navigation } = this.props;
        let imgPath = navigation.getParam('imgPath', 'erro imgPath');

        // Deixando no formato da imagem gerada pelo resultado
        imgPath = imgPath.replace('.png','_output.png');

        this.setState({imgPath: imgPath});

    }

    render () {

        return (

            <View>

                <Image
                    style = {styles.image}
                    source = {{uri: `${urlGetImagem}?nomeImg=${this.state.imgPath}&nomeApp=eFarmer`}}
                    resizeMode = 'contain'
                />

            </View>

        );

    }

}

export default Visualiza;

const styles = StyleSheet.create({ 

    visualizaContainer: {
        justifyContent: 'center',
        alignItems: 'center',
        width: '90%',
        height: '90%',
    },
    image: {
        width: '100%',
        height: '100%',
    },
    

});


