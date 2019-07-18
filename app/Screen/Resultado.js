import React, { Component } from 'react';
import {
    ActivityIndicator,
    View,
    StyleSheet,
    Text, 
    TouchableOpacity
  } from 'react-native';

import axios from 'axios';

import { URL_API } from '../Utils/url_api';

// const urlGetPythonLerImg = `http://192.168.0.160:8080/api/python/processarDados_eFarmer`;
const urlGetPythonLerImg = `${URL_API}/python/processarDados_eFarmer`;

const urlGetImagem = `${URL_API}/imagem/search/findByPath`;


class Resultado extends Component {

    static navigationOptions = {
        title: 'Resultado',
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
        nomeUsuarioLogado: '',
        nomeCompletoLogado: '',
        image: '',
        imagePath: '',
        diagnostico: '',
        resultado: false,
        textoPython: '',
        rotulo: '',
        rotulo_2: '',
        confiRotulo: '',
        confiRotulo_2: '',
        urlPatch: ''
    };

    /**
     * Método para pegar os dados vindo de outra tela e processar os dados
     * @author Pedro Biasutti
     */
    async componentDidMount() {

        const { navigation } = this.props;

        const nomeUsuario = navigation.getParam('nomeUsuario', 'erro nome usuario');
        const image = navigation.getParam('image', 'erro image');
        const imagePath = navigation.getParam('imagePath', 'erro imagePath');


        // console.log('nomeUsuario', nomeUsuario);
        // console.log('image', image);
        // console.log('imagePath', imagePath);

        this.setState({
            nomeUsuarioLogado: nomeUsuario,
            image: image,
            imagePath: imagePath
        });

        this.processarDados(imagePath);
        
    };

    /**
     * Método para chamar o script em python que le um arquivo .png correspondente a imagem desejada e processa os dados.
     * @author Pedro Biasutti
     * @param imagePath - path da imagem
     */
    processarDados = async (imagemPath) => {

        let file = imagemPath;

        let dataStr = '';

        await axios({
            method: 'get',
            url: urlGetPythonLerImg,
            params: {
                path: file
            }
        })
        .then (function(response) {
            console.log('NÃO DEU ERRO PROCESSAR DADOS');
            // console.warn(response.status);
            // console.warn(response.data);
            dataStr = response.data;

        })
        .catch (function(error){
            console.log('DEU ERRO PROCESSAR DADOS');
            // console.warn(error);
            // console.warn(error.request.status);            
        })

        if ( dataStr !== '') {

            var texto = dataStr;

            console.log('texto', texto);

            this.setState({resultado: true, textoPython: texto});

        } else {

            console.log('DEU ERRO NO PYTHON');

        }
    };


    render () {

        return (

            <View style = {styles.viewContainer}>

                {this.state.resultado && 

                    <View>

                        <Text style = {styles.text}>
                            {this.state.textoPython}
                        </Text>

                        <View>
                            {/* <StyledButton onPress = {() => { this.props.navigation.navigate('Menu') }}>
                                Menu
                            </StyledButton> */}

                            <TouchableOpacity 
                                style = {styles.button}
                                onPress = {() => { this.props.navigation.navigate('Menu') }}
                            >
                                <Text style = {styles.textButton}>Menu</Text>
                            </TouchableOpacity>

                        </View>

                    </View>

                }   

                {!this.state.resultado &&

                    <View style = {styles.activity}>
                        <ActivityIndicator/>
                    </View>

                }

            </View>
            
        );
    }
}

export default Resultado;

const styles = StyleSheet.create({ 

    resultadoContainer: {
        justifyContent: 'center',
        alignItems: 'center',
    },
    text: {
        textAlign: 'center',
        fontSize: 16,
        marginBottom: '10%'
    },
    activity: {
        alignItems: 'center',
        transform: ([{ scaleX: 2.5 }, { scaleY: 2.5 }]),
    },
    viewContainer: {
        flex: 1,
        flexDirection: 'column',
        justifyContent: 'space-around',
        alignItems: 'center',
    },
    button: {
        alignSelf: 'stretch',
        backgroundColor: '#39b500',
        borderRadius: 5,
        borderWidth: 1,
        borderColor: '#39b500',
        marginHorizontal: 5,
        marginVertical: 20,
    }, 
    textButton: {
        alignSelf: 'center',
        fontSize: 20,
        fontWeight: '600',
        color: 'white',
        paddingVertical: 10
    }
});

