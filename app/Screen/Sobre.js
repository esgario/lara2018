import React, { Component } from 'react';

import {
        StyleSheet,
        View,
        Alert,
        ScrollView,
        Image
    } from 'react-native';

import axios from 'axios';
import Markdown, {getUniqueID} from 'react-native-markdown-renderer';

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

    };

    /**
     * Método para pegar a descrição do app logo quando montar a tela
     * @author Pedro Biasutti
     */
    async componentDidMount() {

        let texto = '';
        let httpStatus = 0;

        await axios({
            method: 'get',
            url: urlGetDescricao,
            params: {
                titulo: 'descrição E-Farmer',
            },
            headers: { 
                'Cache-Control': 'no-store',
            }
        })
        .then (function(response) {
            console.log('NÃO DEU ERRO PEGA DESCRIÇÃO');
            texto = response.data.texto;
            httpStatus = response.status;
        })
        .catch (function(error){
            console.log('DEU ERRO PEGA DESCRIÇÃO');
            httpStatus = error.request.status;
        })

        if (httpStatus === 200) {

            this.setState({texto_descricao: texto});

        } else {

            const texto ='ERRO 01: \n\nProblema de comunicação com o servidor.' + 
                        '\n\nCaso o problema persista, favor entrar em contato com a equipe técnica.';
            
            Alert.alert(
                'Atenção',
                texto,
                [             
                    {text: 'Ok', onPress: () => {
                        this.props.navigation.navigate('Menu')
                        }
                    },
                ],
                { cancelable: false }
            );

        }

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

                    <View style = {styles.text_Modal}>

                        <Markdown>{this.state.texto_descricao}</Markdown>

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
    text_Modal: {
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



