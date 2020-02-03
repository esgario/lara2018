import React, { Component } from 'react';

import {
    ActivityIndicator,
    View,    
    Alert,
    StyleSheet,
    TouchableOpacity,
    Text,
    Image,
    Dimensions,
} from 'react-native';

import axios from 'axios';

import * as ImageManipulator from 'expo-image-manipulator';

import { URL_API } from '../Utils/url_api';

// Http request
const urlGetNomeUsuario = `${URL_API}/usuario/search/findByNomeUsuario`;
const urlSalvaImagemServidor = `${URL_API}/imagem/upload`;
const urlSalvaImagemDB = `${URL_API}/imagem`;

// Largura tela
const screenWidth = Math.round(Dimensions.get('window').width);


class Aquisicao extends Component {

    state = {

        image: null,
        enviando: false,
        nomeUsuarioLogado: '',
        nomeCompletoLogado: '',
        urlUsr: '',
        imagePath: '',
        latitude: null,
        longitude: null,

    };

    static navigationOptions = {

        // title: 'Aquisição',
        title: 'Image',
        headerStyle: {
          backgroundColor: '#39b500',
        },
        headerTintColor: 'white',
        headerTitleStyle: {
          fontWeight: 'bold',
          fontSize: 28
        },

    };

    /**
     * Método para checar as permissões de acesso a camera e ao álbum de fotos assim que montar a tela
     * @author Pedro Biasutti
     */
    async componentDidMount() {

        const { navigation } = this.props;
        const nomeUsuario = navigation.getParam('nomeUsuario', 'nomeUsuario erro');
        let image = navigation.getParam('image', null);
        const latitude = navigation.getParam('latitude', null);
        const longitude = navigation.getParam('longitude', null);

        image = await this.gerenciaResizeImg(image);

        this.setState({nomeUsuarioLogado: nomeUsuario, image, latitude, longitude});

    };

    /**
     * Método que retorna o usuário sendo passado seu nome do usuário.
     * @author Pedro Biasutti
     * @param nomeUsuario - nome do usuário logado
     */
    getUserByNomeUsuario = async (nomeUsuario) => {

        let nomeUsuarioLogado = nomeUsuario;
        let nomeCompleto = '';
        let validation = false;
        let urlUsr = '';

        await axios({
            method: 'get',
            url: urlGetNomeUsuario,
            params: {
                nomeUsuario: nomeUsuarioLogado,
            },
            headers: { 
                'Cache-Control': 'no-store',
            }
        })
        .then (function(response) {
            console.log('NÃO DEU ERRO GET USUARIO');
            urlUsr = response.data._links.self.href;
            nomeCompleto = response.data.nomeCompleto;
            validation = true;
        })
        .catch (function(error){
            console.log('DEU ERRO GET USUARIO');
        })

        if (validation) {

            this.setState({nomeCompletoLogado: nomeCompleto, urlUsr: urlUsr});

        } else {

            alert('O nome do usuário não existe no banco de dados.');

        }

    };

    /**
     * Método para gerenciar os casos de mudança da imagem
     * @author Pedro Biasutti
     * @param image - objeto imagem
     */
    gerenciaResizeImg = async (image) => {

        const max_size = 1024;

        if (image.width >= image.heigth){

            if (image.width > max_size ) {

                image = await this.resizeImage(image,max_size,null);        

            } 

        } else {

            if (image.height > max_size ) {

                image = await this.resizeImage(image,null,max_size);        

            }

        }

        return image;

    };

    /**
     * Método para manipular a imagem
     * @author Pedro Biasutti
     * @param image - objeto imagem
     * @param img_width - largura de imagem desejada
     * @param img_height - altura de imagem desejada
     */
    resizeImage = async (image, img_width, img_height) => {

        let size = {};
        
        if (img_height != null){

            size['height'] = img_height;

        }

        if (img_width != null){

            size['width'] = img_width;

        }

        const  new_img = await ImageManipulator.manipulateAsync(image.uri, 
            [ { resize: size } ]);

        return new_img;

    };
    
    /**
     * Método para fazer gerenciar o upload da imagem tanto no sevidor quanto no banco de dados.
     * @author Pedro Biasutti
     */
    uploadimage = async () => {

        await this.getUserByNomeUsuario(this.state.nomeUsuarioLogado);

        let urlUsr = this.state.urlUsr;
        let nomeCompleto =  this.state.nomeCompletoLogado;
        let data = new Date(); 

        if ( this.state.image !== null) {

            // Gera o path para salvar as imagens
            let imagePath = nomeCompleto.split(' ').join('') + '-' 
                    + urlUsr.split('//').join('/').split('/').slice(3,5).join('_') + '-' 
                    + data.toString().split(' ').join('_').replace('(','').replace(')','') + '.png';

            this.setState({enviando: true, imagePath});

            let statusSalvaImagemServidor = false;

            // Salva imagem no servidor e pega o status do retorno
            statusSalvaImagemServidor = await this.salvaImagemServidor();

            if (statusSalvaImagemServidor) {

                let statusSalvaImagemBanco = false;
                let urlImg = '';

                // Salva imagem no banco e pega o status do retorno
                [statusSalvaImagemBanco, urlImg] = await this.salvaImagemBanco(data);

                if ( statusSalvaImagemBanco && statusSalvaImagemServidor ) {

                    let image = this.state.image;                    
        
                    this.setState({ 
                                    image: null,
                                    enviando: false,
                                    urlImg,
                                    urlUsr
                    });
        
                    let statusLinkaImagem = false;

                    // Trocando http por https
                    urlImg.includes('http://') ? urlImg = urlImg.replace('http://','https://') : urlImg = urlImg;
                    urlUsr.includes('http://') ? urlUsr = urlUsr.replace('http://','https://') : urlUsr = urlUsr;

                    // Linka o usuário com a imagem
                    statusLinkaImagem = await this.linkaImagem(urlImg, urlUsr);
        
                    if (statusLinkaImagem) {

                        var texto = 'Imagem enviada com sucesso.\n\n' +
                                'Aperte "Ok" para processar os dados!\n\n';
                        
                        // Caso confirmado, vai para pagina de resultados
                        Alert.alert(
                            'Atenção',
                            texto,
                            [             
                                {text: 'Ok', onPress: () => {
                                    this.props.navigation.navigate('Resultado', {
                                        nomeUsuario: this.state.nomeUsuarioLogado,
                                        image,
                                        imagePath: this.state.imagePath
                                    })
                                    }
                                },
                            ],
                            { cancelable: false }
                        );
        
                    } else {

                        this.geraAlerta('ERRO 01: \n\nImagem não pode ser enviada.\n\nCaso o problema persista, favor entrar em contato com a equipe técnica.');
    
                        this.setState({ enviando: false });
    
                    }                     
        
                } else {

                    this.geraAlerta('ERRO 02: \n\nImagem não pode ser enviada.\n\nCaso o problema persista, favor entrar em contato com a equipe técnica.');

                    this.setState({ enviando: false });

                } 

            } else {

                this.geraAlerta('ERRO 03: \n\nImagem não pode ser enviada.\n\nCaso o problema persista, favor entrar em contato com a equipe técnica.');

                this.setState({ enviando: false });

            } 
            
        } 
  
    };

    /**
     * Método para salvar a imagem no servidor
     * @author Pedro Biasutti
     */
    salvaImagemServidor = async () => {

        let status = false;
        let dataForm = new FormData();

        dataForm.append('imagens', {
            name: 'dataHoje',
            type: 'image/jpg',
            uri: this.state.image.uri
            }
        );
        dataForm.append('nomeImg', this.state.imagePath);
        dataForm.append('nomeApp', 'eFarmer');

        await axios({
            method: 'post',
            url: urlSalvaImagemServidor,
            data: dataForm,
            config: { 
                'Content-Type': 'multipart/form-data',
            },
            headers: { 
                'Cache-Control': 'no-store',
            }
        })
        .then (response => {
            console.log('NÃO DEU ERRO SALVA IMAGEM SERVIDOR');
            status = true;
        })
        .catch (error => {
            console.log('DEU ERRO SALVA IMAGEM SERVIDOR');
        })

        return status;

    };

    /**
     * Método para salvar imagem no banco de dados
     * @author Pedro Biasutti
     * @param data - data em que a imagem foi enviada ao app
     */

    salvaImagemBanco = async (data) => {

        let status = false;

        // Arredonda para 6 casas decimais
        const latitude = (Math.round(1000000*parseFloat(this.state.latitude))/1000000).toString()
        const longitude = (Math.round(1000000*parseFloat(this.state.longitude))/1000000).toString()

        await axios({
            method: 'post',
            url: urlSalvaImagemDB,
            data: {
                path: this.state.imagePath,
                rotulo: '',
                confiRotulo: 0.0,
                data: data,
                localizacao: latitude + ',' + longitude,
                app: 'E-Farmer' 
            },
            headers: { 
                'Cache-Control': 'no-store',
            }
        })
        .then (response => {
            console.log('NÃO DEU ERRO SALVAR IMAGEM NO BANCO'); 
            status = true;
            urlImg = response.data._links.imagem.href;
        })
        .catch (error => {
            console.log('DEU ERRO SALAVR IMAGEM NO BANCO');
        })
        
        return [status, urlImg];
    };

    /**
     * Método para linkar a imagem com usuário correspondente.
     * @author Pedro Biasutti
     * @param urlImg - url que aponta para a imagem correspondente.
     * @param urlUsr - url que aponta para o usuário da imagem correspondente.
    */
   linkaImagem = async (urlImg, urlUsr) => {

        let status = false;

        await axios({
            method: 'put',
            url: `${urlImg}/usuario`,
            data: `${urlUsr}`,
            headers: { 
                'Content-Type': 'text/uri-list',
                'Cache-Control': 'no-store',
            }
        })
        .then (function(response) {
            console.log('NÃO DEU ERRO LINKA IMAGEM');
            status = true;
        })
        .catch (function(error){
            console.log('DEU ERRO LINKA IMAGEM');
            console.log(error);
        })
        
        return status;

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
        
        let { image } = this.state;
        let { enviando } = this.state;

        return (

            <View style = {styles.viewContainer}>

                {image &&

                    <Image 
                        source = {{ uri: image.uri }} 
                        style = {{width: '85%', height: '85%'}}
                        resizeMode = 'contain' 
                    />

                }

            {enviando ? (
                
                <View style= {styles.activityIndicator}>
                    <ActivityIndicator/>
                </View>
                
                ) : (

                image &&
                
                <View style = {styles.buttonContainer}>

                    <TouchableOpacity 
                        style = {styles.button}
                        onPress = {this.uploadimage}
                    >
                        <Text style = {styles.buttonText}>Upload</Text>
                    </TouchableOpacity>

                </View>
            )}
                    
            </View>
          
        );

    };

};

export default Aquisicao;

const styles = StyleSheet.create({ 
    buttonContainer: {
        flexDirection: 'column',
        flex: 1,
        width: '50%'
    },
    button: {
        alignSelf: 'stretch',
        backgroundColor: '#39b500',
        borderRadius: 5,
        borderWidth: 1,
        borderColor: '#39b500',
    }, 
    buttonText: {
        alignSelf: 'center',
        color: 'white',
        fontSize: 20,
        fontWeight: '600',
        paddingVertical: 10,
    },
    circularButton: {
        alignItems: 'center',
        justifyContent: 'center',
        backgroundColor: '#e2e2e2',
        height: screenWidth * 0.4,
        width: screenWidth * 0.4,
        borderRadius: screenWidth * 0.4,
        borderWidth: 1,
        borderColor: '#d2d2d2',
        marginHorizontal: 5,
        paddingBottom: 5,
    },
    viewContainer: {
        flex: 1,
        flexDirection: 'column',
        justifyContent: 'space-between',
        alignItems: 'center',
    },
    activityIndicator: {
        marginBottom: '10%',
        transform: ([{ scaleX: 1.5 }, { scaleY: 1.5 }]),
    }
    
});