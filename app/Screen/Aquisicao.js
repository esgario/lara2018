import React, { Component } from 'react';

import {
    ActivityIndicator,
    View,    
    Alert,
    StyleSheet,
    TouchableOpacity,
    Text,
    Image,
    Dimensions
} from 'react-native';

import axios from 'axios';
import { URL_API } from '../Utils/url_api';


const urlGetNomeUsuario = `${URL_API}/usuario/search/findByNomeUsuario`;
const urlPostUpload = `${URL_API}/imagem/upload`; // Salva a imagem no servidor
const urlPost = `${URL_API}/imagem`; // Salva o modelo imagem, definido para o banco, no banco de dados

const screenWidth = Math.round(Dimensions.get('window').width);
const screenHeight = Math.round(Dimensions.get('window').height);

class Aquisicao extends Component {

    state = {
        image: null,
        enviando: false,
        nomeUsuarioLogado: '',
        nomeCompletoLogado: '',
        urlPut: '',
        imagePath: '',
        latitude: null,
        longitude: null,
    };

    static navigationOptions = {
        title: 'Aquisição',
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
        const image = navigation.getParam('image', null);
        const latitude = navigation.getParam('latitude', null);
        const longitude = navigation.getParam('longitude', null);

        this.setState({nomeUsuarioLogado: nomeUsuario, image, latitude, longitude});

    }

    /**
     * Método que retorna o usuário sendo passado seu nome do usuário.
     * @author Pedro Biasutti
     * @param nomeUsuario - nome do usuário logado
     */
    getUserByNomeUsuario = async (nomeUsuario) => {

        let nomeUsuarioLogado = nomeUsuario;
        let nomeCompleto = '';
        let validation = 0;
        let urlPut = '';

        await axios({
            method: 'get',
            url: urlGetNomeUsuario,
            params: {
                nomeUsuario: nomeUsuarioLogado,
            }
        })
        .then (function(response) {
            // console.warn(response.data);
            console.log('NÃO DEU ERRO NO GET USUARIO');
            urlPut = response.data._links.self.href;
            nomeCompleto = response.data.nomeCompleto;
            validation = 7;
        })
        .catch (function(error){
            // console.warn(error);
            console.log('DEU ERRO NO GET USUARIO');
        })

        if ( validation === 7 ) {

            this.setState({
                nomeCompletoLogado: nomeCompleto,
                urlPut: urlPut
            });

        } else {

            alert('O nome do usuário não existe no banco de dados.');

        }

    };
    
    /**
     * Método para fazer o upload da imagem tanto no sevidor quanto no banco de dados.
     * Além disso, salva um arquivo .txt com os pontos selecionados na imagem
     * @author Pedro Biasutti
     */
    uploadimage = async () => {

        await this.getUserByNomeUsuario(this.state.nomeUsuarioLogado);

        let httpStatusServer = 0;
        let httpStatusDb = 0; // Db - Database
        let urlPut = this.state.urlPut;
        let nomeCompleto =  this.state.nomeCompletoLogado;
        let nomeUsuario =  this.state.nomeUsuarioLogado;

        let data = new Date(); 
                
        // Ver se é mais interessante salavr como Date ou como uma String
        // O problema que eu penso em relação a salvar como String é conseguir fazer o filtro depois
        // Já em relação ao Date, no banco pelo menos, não informa as horas

        // data = data.toString();

        if ( this.state.image !== null) {

            this.setState({enviando: true});

            let dataForm = new FormData();
            let pathSalvaImg = nomeCompleto.split(' ').join('') + '-' 
                    + urlPut.split('//').join('/').split('/').slice(3,5).join('_') + '-' 
                    + data.toString().split(' ').join('_').replace('(','').replace(')','') + '.png';

            this.setState({imagePath: pathSalvaImg})

            dataForm.append('imagens', {
                name: 'dataHoje',
                type: 'image/jpg',
                uri: this.state.image.uri
                }
            );
            dataForm.append('nomeImg', pathSalvaImg)
            dataForm.append('nomeApp', 'eFarmer')

            // O primeiro axios fará um post para salvar a imagem no servidor

            await axios({
                method: 'post',
                url: urlPostUpload,
                data: dataForm,
                config: { 
                    'Content-Type': 'multipart/form-data',
                }
            })
            .then (response => {
                console.log('NÃO DEU ERRO 1 AXIOS UPLOAD IMAGEM');
                // console.warn(response.status);
                // console.log('Http status: ',response.status);
                httpStatusServer = response.status;                 
            })
            .catch (error => {
                console.log('DEU ERRO 1 AXIOS UPLOAD IMAGEM');
                // console.warn(error);
                httpStatusServer = error.request.status;
            })

            // O segundo axios fará um post para salvar a imagem no banco de dados

            if ( httpStatusServer == 200 ) {

                // Arredonda para 6 casas decimais
                const latitude = (Math.round(1000000*parseFloat(this.state.latitude))/1000000).toString()
                const longitude = (Math.round(1000000*parseFloat(this.state.longitude))/1000000).toString()

                await axios({
                    method: 'post',
                    url: urlPost,
                    data: {
                        path: pathSalvaImg,
                        segmentado: 'false',
                        rotulo: 'nenhum',
                        confiRotulo: 0.0,
                        rotulo_2: 'nenhum',
                        confiRotulo_2: 0.0,
                        data: data,
                        localizacao: latitude + ',' + longitude 
                    }
                })
                .then (response => {
                    console.log('NÃO DEU ERRO 2 AXIOS UPLOAD IMAGEM');
                    // console.warn(response.status);
                    // console.warn(response.data);
                    httpStatusDb = response.status;
                    urlImg = response.data._links.imagem.href;
                })
                .catch (error => {
                    console.log('DEU ERRO 2 AXIOS UPLOAD IMAGEM');
                    // console.warn(error);
                    httpStatusDb = error.request.status;
                })
            }
            
        }

        if ( httpStatusDb == 201 && httpStatusServer == 200 ) {

            let image = this.state.image;

            this.setState({ 
                            imgWidth: this.state.image.width,
                            image: null,
                            enviando: false,
                            nomeCompletoLogado: nomeCompleto,
                            nomeUsuarioLogado: nomeUsuario,
                            urlImg: urlImg,
                            urlPut: urlPut
            });


            httpStatusLinkaImagem = await this.linkaImagem(urlImg, urlPut);

            if (httpStatusLinkaImagem === 204) {
            // if (true) { // Serve apenas para testar no emulador

                var texto = 'Imagem enviada com sucesso.\n\n' +
                        'Aperte "Ok" para processar os dados!\n\n';

                Alert.alert(
                    'Atenção',
                    texto,
                    [             
                        {text: 'Ok', onPress: () => {
                            this.props.navigation.navigate('Resultado', {
                                nomeUsuario: this.state.nomeUsuarioLogado,
                                image: image,
                                imagePath: this.state.imagePath
                            })
                            }
                        },
                    ],
                    { cancelable: false }
                );

            }

            

        } else {
            alert('Imagem não pode ser enviada.\n\nCaso o problema persista, favor entrar em contato com a equipe técnica.');
        }  
    };

    /**
     * Método para linkar a imagem com usuário correspondente.
     * @author Pedro Biasutti
     * @param urlImg - url que aponta para a imagem correspondente.
    */
   linkaImagem = async (urlImg, urlPut) => {

        await axios({
            method: 'put',
            url: `${urlImg}/usuario`,
            data: `${urlPut}`,
            headers: { 
                'Content-Type': 'text/uri-list',
            }
        })
        .then (function(response) {
            console.log('NÃO DEU ERRO LINKA IMAGEM');
            // console.warn(response.status);
            // console.log('Http status: ',response.status);
            httpStatus = response.status; // 204
        })
        .catch (function(error){
            console.log('DEU ERRO LINKA IMAGEM');
            // console.warn(error);
        })
        
        return httpStatus;
    };
    
    render () {
        
        let { image } = this.state;
        let { enviando } = this.state;

        return (

        <View style = {styles.viewContainer}>

                {image &&

                    <Image 
                        source = {{ uri: image.uri }} 
                        style = {styles.image} 
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
    }
}

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
        marginHorizontal :5,
        marginVertical: 20,
    }, 
    buttonText: {
        alignSelf: 'center',
        color: 'white',
        fontSize: 20,
        fontWeight: '600',
        paddingVertical: 10,
    },
    image: {
        width: 0.95 * screenWidth,
        height: 0.65 * screenHeight
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
        marginTop: 50
    },
    activityIndicator: {
        marginBottom: 15,
        transform: ([{ scaleX: 1.5 }, { scaleY: 1.5 }]),
    }
});