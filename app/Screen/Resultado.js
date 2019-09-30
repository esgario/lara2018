import React, { Component } from 'react';
import {
    ActivityIndicator,
    View,
    StyleSheet,
    Text, 
    TouchableOpacity,
    Alert,
    Image,
    ScrollView,
    Dimensions
} from 'react-native';

import axios from 'axios';

import { URL_API } from '../Utils/url_api';

import Markdown, {getUniqueID} from 'react-native-markdown-renderer';

// Http request
const urlGetImagem = `${URL_API}/imagem/baixar`;
const urlPegaModeloResultado = `${URL_API}/resultado/pegaModeloResultado`;
const urlCriaRequi = `${URL_API}/python/inference`;
const urlChecaResultado = `${URL_API}/python/result`;

// Dimensões da tela
const screenWidth = Math.round(Dimensions.get('window').width);
const screenHeigth = Math.round(Dimensions.get('window').height);

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
        uriImg: '',
        resultado: false,
        textoPython: '',
        textoModelo: '',
        result_status: ''

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

        this.setState({
            nomeUsuarioLogado: nomeUsuario,
            image: image,
            imagePath: imagePath
        });

        this.criaRequisicao(imagePath);
        
    };

    /**
     * Método para enviar os dados necessários para processar para que seja gerado um job_id
     * @author Pedro Biasutti
     * @param imagePath - path da imagem
     */
    criaRequisicao = async (imagemPath) => {

        console.log('criaRequisicao');
        console.log('imagemPath',imagemPath);

        let resposta = 'error';
        
        await axios({
            method: 'get',
            url: urlCriaRequi,
            params: {
                path: imagemPath,
                algoritmo: 'coffee'
            }
        })
        .then (function(response) {
            console.log('response.data',response.data);
            console.log('NÃO DEU ERRO CRIA REQUISIÇÃO');
            resposta = response.data;
        })
        .catch (function(error){
            console.log(error);
            console.log('DEU ERRO CRIA REQUISIÇÃO');     
        })

        if ( resposta !== 'error') {

            console.log('resposta',resposta);
            this.gerenciaChecaResult(resposta);

        } else {

            console.log('DEU ERRO RESPOSTA CRIA REQUISIÇÃO'); 

        }

    };

    /**
     * Método para chamar o script em python que le um arquivo .png correspondente a imagem desejada e processa os dados.
     * @author Pedro Biasutti
     * @param imagePath - path da imagem
     */
    checaResultado = async (job_id) => {

        let resp = 'error';        

        console.log('veio no checaResult');

        await axios({
            method: 'get',
            url: urlChecaResultado,
            params: {
                job_id: job_id
            }
        })
        .then (function(response) {
            console.log('NÃO DEU CHECA RESULTADO');
            console.log('response.data',response.data);
            resp = response.data;
        })
        .catch (function(error){
            console.log('DEU ERRO CHECA RESULTADO');           
        })

        // if ( resp !== 'error' && resp !== '') {
        if ( resp !== 'error') {

            if (resp == 'processando') {

                console.log(resp);
                this.setState({result_status: resp});

            } else {

                outImgPath = this.state.imagePath.replace('.png','_output.png');
                uriImg = `${urlGetImagem}?nomeImg=${outImgPath}&nomeApp=eFarmer`;

                console.log('resp', resp);
                console.log('uriImg',uriImg);
    
                this.setState({result_status: 'completo', textoPython: resp, uriImg: uriImg});

                console.log('this.state.textoPython 001', this.state.textoPython);
                console.log('this.state.uriImg 001', this.state.uriImg);

            }

        } else {

            console.log('DEU ERRO RESPOSTA CHECA RESULTADO');

        }

    };

    gerenciaChecaResult = async (job_id) => {
        
        let count = 0;
        let result = '';

        while (count <10 && this.state.textoPython == ''){

            console.log(`Tentativa numero: ${count}`);

            await this.checaResultado(job_id);

            // Função para forcar delay
            await this.performTimeConsumingTask(3000);

            count = count + 1;

        }

        if (count = 10 && this.state.textoPython == '') {

            let texto = 'O número de tentativas excedeu o permitido. Tente novamente'


            // Caso confirmado, vai para pagina o menu
            Alert.alert(
                'Atenção',
                texto,
                [             
                    {text: 'Ok', onPress: () => {
                        this.props.navigation.navigate('Menu', {
                            nomeUsuario: this.state.nomeUsuarioLogado,
                        })
                        }
                    },
                ],
                { cancelable: false }
            );

        }

        result = this.state.textoPython;
        console.log('this.state.textoPython',result);
        
        this.analizaResposta(result);

    };

    /**
     * Método para fazer o delay dado tempo em milissegundos
     * @author Pedro Biasutti
     * @param time - milissegundos
     */
    performTimeConsumingTask = async(time) => {

        return new Promise((resolve) =>
        
            setTimeout(
                () => { resolve('result') },
            time
            )

        );

    }

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

        console.log(modeloResp);

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
     * Método para exibir um alerta customizado
     * @author Pedro Biasutti
     */
    geraAlerta = (textoMsg) => {

        var texto = textoMsg

        Alert.alert(
            'Atenção',
            texto,
            [
                {text: 'OK'},
              ],
            { cancelable: false }
        );
        
    };

    render () {

        let { resultado } = this.state;

        return (

            <View style = {styles.resultadoContainer}>

                {resultado ? 
                
                    (
                        <ScrollView>

                            <View style = {styles.resultadoContainer}>

                                <View style = {styles.imageView}>

                                    <Image
                                        source = {{uri: this.state.uriImg}}
                                        resizeMode = 'contain'
                                        style = {styles.image}  
                                    />

                                </View>

                                    <View style = {styles.markdown}>

                                        <ScrollView>

                                            <Markdown rules={rules}>{this.state.textoModelo}</Markdown>

                                        </ScrollView>

                                    </View>

                                <TouchableOpacity
                                    style = {styles.button}
                                    onPress = {() => { this.props.navigation.navigate('Menu') }}
                                >

                                    <Text style = {styles.textButton}>Menu</Text>

                                </TouchableOpacity>

                            </View>

                        </ScrollView>                            

                    ) : (

                        <View style = {styles.activity}>

                            <ActivityIndicator/>

                        </View>

                    )

                }

            </View>

        );

    };
    
};

export default Resultado;

const styles = StyleSheet.create({ 

    resultadoContainer: {
        flex: 1,
        flexDirection: 'column',
        justifyContent: 'center',
        alignItems: 'center',
    },
    text: {
        textAlign: 'center',
        fontSize: 16,
        marginBottom: '10%'
    },
    activity: {
        flex: 1,
        flexDirection: 'column',
        justifyContent: 'center',
        alignItems: 'center',
        transform: ([{ scaleX: 2.5 }, { scaleY: 2.5 }]),
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
    },
    markdown:{
        flexDirection: 'row',
        marginVertical: 10,
        paddingHorizontal: 10,
        height: 0.20 * screenHeigth,
    },
    image: {
        width: 0.9 * screenWidth,
        height: 0.9 * screenWidth,
    },
    imageView: {
        marginVertical: 10
    }
    
});

