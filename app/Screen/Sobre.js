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
const urlGetDescricao = `${URL_API}/sobre/search/findByTitulo`;

class Sobre extends Component {

    static navigationOptions = {

        title: 'Sobre',
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
        texto_descricao: null
    }

    /**
     * Método para pegar a descrição do app logo quando montar a tela
     * @author Pedro Biasutti
     */
    async componentDidMount() {

        texto = '';

        await axios({
            method: 'get',
            url: urlGetDescricao,
            params: {
                titulo: 'descrição',
            }
        })
        .then (function(response) {
            console.log('NÃO DEU ERRO PEGA DESCRIÇÃO');
            texto = response.data.texto;

        })
        .catch (function(error){
            console.log('DEU ERRO PEGA DESCRIÇÃO');       
        })

        this.setState({texto_descricao: texto});

    };

    render () {

        return (

            <ScrollView>

                <View style = {styles.sobreContainer}> 

                    <View style = {styles.imageContainer}>

                        <Image
                            style = {styles.image}
                            source = {require('./../assets/labcin.png')}
                            resizeMode = 'contain'
                        />

                    </View>

                    <View>

                        <Text style = {styles.text}>{this.state.texto_descricao}</Text>

                    </View>

                </View>

            </ScrollView>
        );

    };

};

export default Sobre;

const styles = StyleSheet.create({
    sobreContainer: {
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



