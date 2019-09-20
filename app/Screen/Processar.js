import React, { Component } from 'react';

import {
        StyleSheet,
        View,
        Text,
        ScrollView,
        Image
    } from 'react-native';

import axios from 'axios';

import { URL_API } from '../Utils/url_api';

// Http request
const urlGetDescricao = `${URL_API}/Processar/search/findByTitulo`;

class Processar extends Component {

    static navigationOptions = {

        title: 'Processar',
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
        nomeUsuario: '',
        image: null,
        imagePath: ''
    }

    /**
     * Método para pegar a descrição do app logo quando montar a tela
     * @author Pedro Biasutti
     */
    async componentDidMount() {

        const { navigation } = this.props;
        const nomeUsuario = navigation.getParam('nomeUsuario', 'nomeUsuario erro');
        const image = navigation.getParam('image', null);
        const imagePath = navigation.getParam('imagePath', 'erro imagePath');

        this.setState({nomeUsuarioLogado: nomeUsuario, image, imagePath});


    };

    render () {

        return (



                <View style = {styles.ProcessarContainer}>

                    <Text>{this.state.nomeUsuario}</Text> 
                    <Text>{this.state.imagePath}</Text> 

                </View>

        );

    };

};

export default Processar;

const styles = StyleSheet.create({
    ProcessarContainer: {
        justifyContent: 'center',
        alignItems: 'center'
    },
    text: {
        textAlign: 'justify',
        color: 'black',
        fontSize: 16,
        fontWeight: '400',
        lineHeight: 20,
        marginHorizontal: 10
    },
    image: {
        width: 109, 
        height: 150
    },
    imageContainer: {
        marginVertical: 20,
    }
});



