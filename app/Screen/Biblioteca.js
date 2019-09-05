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
    FlatList,
    Modal
} from 'react-native';

import axios from 'axios';
import Icon from 'react-native-vector-icons/FontAwesome';
import { URL_API } from '../Utils/url_api';

import { GestureHandler } from 'expo';
const { Swipeable } = GestureHandler;

// Http request
const urlGetNomeUsuario = `${URL_API}/usuario/search/findByNomeUsuario`;
const urlGetImagem = `${URL_API}/imagem/baixar`;
const urlGetImagemByPath = `${URL_API}/imagem/search/findByPath`;

// Largura tela
const screenWidth = Math.round(Dimensions.get('window').width);

class Biblioteca extends Component {

    state = {
        user: '',
        nomeUsuarioLogado: '',
        imagens: [],
        urlImagem: '',
        urlUser: '',
        deletando: false,
        exibeImagem : false,
        uriImagemModal: '',
        imagensDisplay: [],
        imagenSizeDisplay: 4, // Controla quantas imagens devem ser exiidas por vez
        imagemIndexDisplay: 0, // Controla o index das imagens exibidas
    };

    static navigationOptions = {
        title: 'Biblioteca',
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
     * Método para pegar o nome do usuário passado da tela anterior e chamar método para pega-lo
     * @author Pedro Biasutti
     */
    async componentDidMount() {

        const { navigation } = this.props;
        const nomeUsuario = navigation.getParam('nomeUsuario', 'nomeUsuario erro');

        this.setState({nomeUsuarioLogado: nomeUsuario});

        this.getUserByNomeUsuario(nomeUsuario);

        console.log('nomeUsuario',nomeUsuario);
 
    };

    /**
     * Método que retorna o usuário sendo passado seu nome do usuário.
     * @author Pedro Biasutti
     * @param nomeUsuario - nome do usuário logado
     */
    getUserByNomeUsuario = async (nomeUsuario) => {

        let urlUser = '';
        let validation = 0;
        let data;

        await axios({
            method: 'get',
            url: urlGetNomeUsuario,
            params: {
                nomeUsuario: nomeUsuario,
            }
        })
        .then (function(response) {
            data = response.data
            urlUser = response.data._links.self.href;
            validation = 7;
            console.log('NÃO DEU ERRO NO GET USUARIO');
        })
        .catch (function(error){
            console.log('DEU ERRO NO GET USUARIO');
        })

        if ( validation === 7 ) {

            if (data.imagem.length) {

                let fim = 0;
                let imgDisp = [];
                let iniLoop = this.state.imagemIndexDisplay;
                let fimLoop = this.state.imagenSizeDisplay;

                fim = iniLoop + fimLoop;

                if (fimLoop > data.imagem.length) {

                    fimLoop = data.imagem.length;

                }

                for (var i = iniLoop; i < fimLoop; i++) {
                    imgDisp.push(data.imagem[i]);
                }

                this.setState({
                    user: data,
                    imagens: data.imagem,
                    urlUser: urlUser,
                    imagensDisplay: imgDisp,
                    imagemIndexDisplay: fim,
                });

            } else {

                var texto = 'Não existem imagens a serem exibidas.\n\n' +
                        'Aperte "Ok" para voltar ao Menu inicial!\n\n';

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

        } else {

            alert('O nome do usuário não existe no banco de dados !');

        }

    };

    /**
     * Método para pegar imagem dado o path
     * @author Pedro Biasutti
     * @param path - nome da imagem
     */
    getImagemPorPath = async (path) =>{

        let httpStatus = 0;
        let urlImagem = '';

        await axios({
            method: 'get',
            url: urlGetImagemByPath,
            params: {
                path: path,
            }
        })
        .then (function(response) {
            urlImagem = response.data._links.self.href;
            httpStatus = response.status;
            // console.warn(response.status);
        })
        .catch (function(error){
            // console.warn(error);
            console.log('DEU ERRO NO GET IMAGEM');
        })

        if ( httpStatus === 200 ) {

            this.setState({ urlImagem: urlImagem });

        }

        return httpStatus;

    };

    /**
     * Método para deslinkar a imagem com usuário correspondente.
     * @author Pedro Biasutti
     * @param urlImg - url que aponta para a imagem correspondente.
     * @param urlUser - url que aponta para o usuário correspondente.
    */
   deslinkaImagem = async (urlImg, urlUser) => {

        await axios({
            method: 'delete',
            url: `${urlImg}/usuario`,
            data: `${urlUser}`,
            headers: { 
                'Content-Type': 'text/uri-list',
            }
        })
        .then (function(response) {
            console.log('NÃO DEU ERRO DESLINKA IMAGEM');
            // console.warn(response.status);
            httpStatus = response.status; // 204
        })
        .catch (function(error){
            console.log('DEU ERRO DESLINKA IMAGEM');
            // console.warn(error);
        })
        
        if( httpStatus === 204 ) {

            // Reseta o sistema
            this.setState({ 
                user: '',
                imagens: [],
                urlUser: '',
                imagensDisplay: [],
                imagenSizeDisplay: 4,
                imagemIndexDisplay: 0
            });

        }

        return httpStatus;
    };

    /**
     * Método para desvincular a imagem do usuário.
     * Importante ressaltar que não deleta a imagem, mas o link entre usuário e imagem
     * @author Pedro Biasutti
     * @param path - nome da imagem a ser deletada
     */
    deletaItem = async (path) => {

        let httpStatusGetImagem = 0;
        let httpStatusdeslinkaImagem = 0;

        httpStatusGetImagem = await this.getImagemPorPath(path);

        if ( httpStatusGetImagem === 200) {

            httpStatusdeslinkaImagem = await this.deslinkaImagem(this.state.urlImagem, this.state.urlUser);

            if ( httpStatusdeslinkaImagem === 204 ) {

                await this.getUserByNomeUsuario(this.state.nomeUsuarioLogado);

                alert('A imagem foi deletada com sucesso !');

            } else {

                alert('ERRO 01:\n\nA imagem não pode ser deletada. \n\n Tente novamente, caso o erro persista contate a equipe de suporte.');
    
            }

        } else {

            alert('ERRO 02:\n\nA imagem não pode ser deletada. \n\n Tente novamente, caso o erro persista contate a equipe de suporte.');

        }

    };

    /**
     * Método para carregar as demais imagens para serem exibidas na tela, caso exista
     * @author Pedro Biasutti
     */
    carregaImagem = () => {

        let fim = this.state.imagemIndexDisplay + this.state.imagenSizeDisplay;
        let imgDisp = this.state.imagensDisplay;
        let imgLength = this.state.imagens.length;       
        
        if (fim > imgLength){
            fim = imgLength;
        }

        for (var i = this.state.imagemIndexDisplay; i < fim; i++) {
            imgDisp.push(this.state.imagens[i]);
        }

        this.setState({
            imagensDisplay: imgDisp,
            imagemIndexDisplay: fim
        });

    };

    /**
     * Método para fazer o layout do Flatlist
     * @author Pedro Biasutti
     */
    renderItem = ({item}) => (

        <Swipeable renderRightActions = {() => this.RightActions(item.path)}>

            <View style = {styles.listContainer}>               

                <TouchableOpacity
                    onPress = {() => this.props.navigation.navigate('Visualiza', {imgPath: item.path})}
                >
                    <View style = {styles.listSubContainer}>

                        <View>
                            <TouchableOpacity 
                                onPress = {() => this.setState({ 
                                    exibeImagem: true,
                                    uriImagemModal: `${urlGetImagem}?nomeImg=${item.path}&nomeApp=eFarmer`
                                    }
                                )}
                            >
                                <Image
                                    style = {styles.imageListItSelf}
                                    source = {{uri: `${urlGetImagem}?nomeImg=${item.path}&nomeApp=eFarmer&largura=100&altura=100`}}
                                />

                            </TouchableOpacity>                        
                        </View>

                        <View>

                            <View style = {styles.imageTextContainer}>
                                <Text style = {styles.imageTextTitle}>Data: </Text>
                                <Text style = {styles.imageTextDescription}>{item.data}</Text>   
                            </View>
                            <View style = {styles.imageTextContainer}>
                                <Text style = {styles.imageTextTitle}>Latitude: </Text>
                                <Text style = {styles.imageTextDescription}>{item.localizacao.split(',')[0]}</Text>   
                            </View>
                            <View style = {styles.imageTextContainer}>
                                <Text style = {styles.imageTextTitle}>Longitude: </Text>
                                <Text style = {styles.imageTextDescription}>{item.localizacao.split(',')[1]}</Text>   
                            </View>

                        </View>

                    </View>

                </TouchableOpacity>

            </View>

        </Swipeable>

    );

    /**
     * Método para fazer o layout do botão direito no flatlist/swipeable.
     * Caso o usuário queira, pode deletar a imagem selecionada
     * @author Pedro Biasutti
     * @param path - nome da imagem a ser deletada
     */
    RightActions = (path) =>(

            <View style = {styles.rigthActions}>
                
                { this.state.deletando ? (
                    <ActivityIndicator/>
                    ) : (
                    <TouchableOpacity 
                        onPress = {() => this.deletaItem(path)}
                    >
                        <Icon name = 'trash' size={32} color="#2b2c2d"/>
                    </TouchableOpacity>
                )}

            </View>

    );
    
    render () {

        return (

        <View style ={styles.container}> 
            
            <FlatList
                contentContainerStyle = {styles.list}
                // data = {this.state.imagens}
                data = {this.state.imagensDisplay}
                keyExtractor = {item => item.path}
                renderItem = {this.renderItem}
                onEndReached = {() => this.carregaImagem()}
            />

            <Modal
                transparent = {true}
                visible = {this.state.exibeImagem}
                onRequestClose = {() => {
                console.log('Modal has been closed.');
                }}
            >   

                <View style={styles.modalContainer}>
                    
                    <Image
                        style = {styles.imageModal} 
                        source = {{uri: this.state.uriImagemModal}}
                        resizeMode = 'contain'
                    />

                    <TouchableOpacity
                        onPress = {() => {
                            this.setState({ exibeImagem: false });
                        }}
                        style = {styles.button}
                    >

                        <Text style = {styles.buttonText}>Fechar</Text>

                    </TouchableOpacity>
                    
                </View>

            </Modal>

        </View>
            
        );

    };

};

export default Biblioteca;

const styles = StyleSheet.create({ 
    container: {
        flex: 1,
        backgroundColor: '#fafafa'
    }, 
    list: {
        padding: 5,
    },
    listContainer: {
        backgroundColor: '#fff',
        padding: 20,
        borderWidth: 1,
        borderColor: 'black',
        borderRadius: 5,
        marginBottom: 10,
    },
    listSubContainer: {
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'space-between'
    },
    imageTextContainer: {
        flexDirection: 'row',
        alignItems: 'center'
    },
    imageTextTitle: {
        fontSize: 15,
        fontWeight: 'bold',
        color: '#333'
    },
    imageTextDescription: {
        fontSize: 13,
        color: '#999',
        lineHeight: 24,
        textAlign: 'justify'
    },
    imageListItSelf: {
        width: 0.30 * screenWidth,
        height: 0.30 * screenWidth
    },
    rigthActions: {
        backgroundColor: 'red',
        justifyContent: 'center',
        alignItems: 'flex-end',
        borderWidth: 1,
        borderColor: 'black',
        borderRadius: 5,
        marginBottom: 10,
        paddingHorizontal: 20,

    },
    button: {
        alignSelf: 'center',
        width: 0.8 * screenWidth,
        backgroundColor: '#e2e2e2',
        borderRadius: 5,
        borderWidth: 1,
        borderColor: '#d2d2d2',
    },
    buttonText: {
        alignSelf: 'center',
        color: 'white',
        fontSize: 20,
        fontWeight: '600',
        paddingVertical:10,
    },
    modalContainer: {
        flex: 1,
        flexDirection: 'column',
        justifyContent: 'center',
        alignItems: 'center',
    },
    imageModal: {
        width: '80%',
        height: '80%',
    }

});
