import React, { Component } from 'react';
import {
        View,
        StyleSheet,
        Image,
        ScrollView,
        ActivityIndicator,
        Dimensions,
        Modal
    } from 'react-native';

import axios from 'axios';

import Markdown, {getUniqueID} from 'react-native-markdown-renderer';

import { URL_API } from '../Utils/url_api';

// Http request
const urlGetImagem = `${URL_API}/imagem/baixar`;
const urlChecaResultado = `${URL_API}/python/checaResultado`;
const urlPegaModeloResultado = `${URL_API}/resultado/pegaModeloResultado`;

// MarkDown rules
const rules = {
    heading1: (node, children, parent, styles) =>
        <Text key={getUniqueID()} style={[styles.heading, styles.heading1]}>
            [{children}]    
        </Text>,
    heading2: (node, children, parent, styles) =>
        <Text key={getUniqueID()} style={[styles.heading, styles.heading2]}>
            [{children}]
        </Text>,
    heading3: (node, children, parent, styles) =>
        <Text key={getUniqueID()} style={[styles.heading, styles.heading3]}>
            [{children}]
        </Text>,
};

// Dimensões da tela
const screenWidth = Math.round(Dimensions.get('window').width);
const screenHeigth = Math.round(Dimensions.get('window').height);

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

        img_path: '',
        textoModelo: '',
        resultado: false,
        img_loading: false

    };

    /**
     * Método para pegar os dados vindo de outra tela e processar os dados
     * @author Pedro Biasutti
     */
    async componentDidMount() {

        const { navigation } = this.props;
        let img_path = navigation.getParam('img_path', 'erro img_path');        

        this.setState({img_path: img_path.replace('.png','_output.png')});

        await this.checaResultado(img_path);

    };

    /**
     * Método para verficar se a imagem no banco possui job_id e em caso afirmativo
     * checar se possui ou não o resultado da classficação. Caso não possua, uma requisição
     * ao servidor em python será feita
     * @author Pedro Biasutti
     */
    checaResultado = async (img_path) => {

        let resposta = 'error';
        
        await axios({
            method: 'get',
            url: urlChecaResultado,
            params: {
                img_path: img_path,
            }
        })
        .then (function(response) {
            console.log('NÃO DEU ERRO CHECA RESULTADO');
            resposta = response.data;
        })
        .catch (function(error){
            console.log('DEU ERRO CHECA RESULTADO');     
        })

        await this.analizaResposta(resposta);

    };

    /**
     * Método para pegar o retorno do codigo que processas o dados e analizar para os 3 tipos de respostas
     * @author Pedro Biasutti
     * @param resp - resposta
     */
    analizaResposta = async (resposta) => {

        let resp = resposta;
        resp = resp.split('\n');
        let nomeApp = 'eFarmer';

        // Pega layout do resultado
        modeloResp = await this.pegaModeloResultado(nomeApp);
        aux = modeloResp.split('\n');
        aux = aux[aux.length - 1];

        if ( (resp.length - 1) < 2 ) {

            // Quebra a resposta entre seus itens (numero da lesão, diagnóstico, probabilidade)
            lesaoInfo = resp[0].split(',');

            // Da o replace para colocar os dados reais
            modeloResp = modeloResp.replace('%num', lesaoInfo[0]).replace('%diag', lesaoInfo[1]).replace('%prob', lesaoInfo[2]);

        } else {

            // Aumenta o layout de acordo com a qtd de lesão
            for (x = 0; x< (resp.length - 2); x++) {

                // Quebra a resposta entre seus itens (numero da lesão, diagnóstico, probabilidade)
                lesaoInfo1 = resp[x].split(',');
                lesaoInfo2 = resp[x+1].split(',');

                // Da o replace para colocar os dados reais
                modeloResp = modeloResp.replace('%num', lesaoInfo1[0]).replace('%diag', lesaoInfo1[1]).replace('%prob', lesaoInfo1[2])
                            + '\n' + 
                            aux.replace('%num', lesaoInfo2[0]).replace('%diag', lesaoInfo2[1]).replace('%prob', lesaoInfo2[2]);

            }

        }

        this.setState({resultado: true, textoModelo: modeloResp});

    };

    /**
     * Método para pegar o estilo MarkDown do resultado correspondente a quantidade de parametros de saida do processa dados python
     * @author Pedro Biasutti
     */
    pegaModeloResultado = async (nomeApp) => {

        let modeloResp = '';

        await axios({
            method: 'get',
            url: urlPegaModeloResultado,
            params: {
                nomeApp: nomeApp,
            }
        })
        .then (function(response) {
            console.log('NÃO DEU ERRO PEGA MODELO RESULTADO');
            modeloResp = response.data;

        })
        .catch (function(error){
            console.log('DEU ERRO PEGA MODELO RESULTADO');
        })

        return modeloResp;

    };

    /**
     * Método para pegar o estilo MarkDown do resultado correspondente a quantidade de parametros de saida do processa dados python
     * @author Pedro Biasutti
     */
    renderVisualiza () {

        let { resultado } = this.state;

        if (resultado){

            return (
                
                <View style = {styles.visualizaSubcontainer}>

                    <View style = {styles.imageContainer}>

                        <Image
                            style = {styles.image}
                            source = {{uri: `${urlGetImagem}?nomeImg=${this.state.img_path}&nomeApp=eFarmer&altura=${10*screenHeigth}`}}
                            resizeMode = 'contain'
                            onLoadStart = {() => this.setState({img_loading: true})}
                            onLoadEnd = {() => this.setState({img_loading: false})}
                        />

                    </View>                    

                    <Modal
                        transparent = {true}
                        visible = {this.state.img_loading}
                        onRequestClose = {() => {
                        console.log('Modal has been closed.');
                        }}
                    >

                        <View style = {styles.activity}>

                            <ActivityIndicator/>

                        </View>

                    </Modal>

                    <View style = {styles.markdownContainer}>

                        <ScrollView>

                            <Markdown rules={rules}>{this.state.textoModelo}</Markdown>

                        </ScrollView>

                    </View> 

                </View>

            );

        } else {

            return (

                <View style = {styles.activity}>

                    <ActivityIndicator/>

                </View>

            );

        }


    };

    render () {

        return (

            <View style = {styles.visualizaContainer}>
              
              {this.renderVisualiza()}

            </View>

        );

    };

}

export default Visualiza;

const styles = StyleSheet.create({ 

    visualizaContainer: {
        flex: 1,
        flexDirection: 'column',
        justifyContent: 'center',
        alignItems: 'center',
    },
    visualizaSubcontainer: {
        flex: 1,
        flexDirection: 'column',
        justifyContent: 'center',
        alignItems: 'center',
    },
    imageContainer: {
        flexDirection: 'row',
        height: 0.5 * screenHeigth,
        marginHorizontal: 20,
        marginVertical: 20
    },  
    image: {
        width: '100%',
        height: '100%',
    },
    markdownContainer: {
        flexDirection: 'row',
        marginHorizontal: 20,
        marginBottom: 20,
        height: 0.3 * screenHeigth
    },   
    activity: {
        flex: 1,
        flexDirection: 'column',
        justifyContent: 'center',
        alignItems: 'center',
        transform: ([{ scaleX: 2.5 }, { scaleY: 2.5 }]),
    }
    

});
