import React, { Component } from 'react';

import { StyleSheet,
        Text,
        View, 
        Image, 
        TouchableOpacity, 
        TextInput,
        ActivityIndicator,
        Dimensions,
        Modal,
        ScrollView,
        AsyncStorage,
        Alert
    } from 'react-native';

import { Formik } from 'formik';
import * as yup from 'yup';
import axios from 'axios';

import Markdown, {getUniqueID} from 'react-native-markdown-renderer';
import { CheckBox } from 'react-native-elements'

import { URL_API } from '../Utils/url_api';

// Dimensões da tela
const screenWidth = Math.round(Dimensions.get('window').width);
const screenHeight = Math.round(Dimensions.get('window').height);

// Http request
const urlGetLogIn = `${URL_API}/usuario/verificaLogin?`;
const urlGetTermosUso = `${URL_API}/termos/search/findByTitulo`;
// const urlACK = `${URL_API}/ack/teste`;

// YUP validation
const validationSchema = yup.object().shape({
    nomeUsuario: yup
    .string()
    .required('Nome do usuário não foi informado')
    .label('nomeUsuario'),
    senha: yup
    .string()
    .required('Senha não foi informada')
    .label('senha')
});

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

class Home extends Component {

    static navigationOptions = {

        title: 'Log in',
        headerStyle: {
          backgroundColor: '#39b500',
        },
        headerTintColor: 'white',
        headerTitleStyle: {
          fontWeight: 'bold',
          fontSize: 30
        }

    };

    state = {

        exibe_termos_de_uso: false,
        exibe_instrucoes_de_uso: false,
        texto_termoUso: null,
        checked: false

    };

    /**
     * Método para verificar na memória se as variáveis de log foram salvas, assim que montar a tela
     * @author Pedro Biasutti
     */
    async componentDidMount() {

        let nomeUsuario = '';
        let senha = '';

        // Procura as variáveis na memória
        nomeUsuario = await this._retrieveData('nomeUsuario');
        senha = await this._retrieveData('senha');

        // Se estiver ok quer dizer que o usuário está salvo (e já foi feita validação no servidor)
        if (nomeUsuario !== 'erro' && senha !== 'erro') {

            // Da seguimento no fluxo do app
            this.props.navigation.navigate('Menu', {nomeUsuario: nomeUsuario});

        } else {

            this.setState({exibe_instrucoes_de_uso: true});

        }

    };

    /**
     * Método para checar se tanto o nomeUsuario quanto senha existem no banco de dados e se pertencem ao mesmo usuário.
     * @author Pedro Biasutti
     * @param values - Dados que foram digitados no form.
     */
    validaForm = async (values) => {

        let validation = 0;

        await axios({
            method: 'get',
            url: urlGetLogIn,
            params: {
                nomeUsuario: values.nomeUsuario,
                senha: values.senha
            },
            headers: { 
                'Cache-Control': 'no-store',
            }
        })
        .then (function(response) {
            console.log('NÃO DEU ERRO CHECA NO BANCO');
            validation = response.data;

        })
        .catch (function(error){
            console.log('DEU ERRO CHECA NO BANCO');       
        })

        if ( validation === -1) {

            alert('O usuário informado não foi encontrado.\n\nPor favor, verifique o usuário digitado e tente novamente.');

        } else if ( validation === 1 ){

            alert('Senha incorreta.\n\nPor favor, digite a senha novamente.\n\nCaso o erro persista, solicitar nova senha');

        }

        return validation;

    };

    /**
     * Método para buscar os termo de uso do app no banco e habilitar exibição no Modal
     * @author Pedro Biasutti
     */
    temosUso = async () => {

        texto = '';

        await axios({
            method: 'get',
            url: urlGetTermosUso,
            params: {
                titulo: 'Termos de uso E-Farmer',
            },
            headers: { 
                'Cache-Control': 'no-store'
            }
        })
        .then (function(response) {
            console.log('NÃO DEU ERRO PEGA TERMOS DE USO');
            texto = response.data.texto;

        })
        .catch (function(error){
            console.log('DEU ERRO PEGA TERMOS DE USO');
            // console.warn(error);
            // console.warn(error.status);
            texto = error.status;
        })

        if (texto !== undefined) {
            
            this.setState({texto_termoUso: texto, exibe_termos_de_uso: true});

        } else {

            this.geraAlerta('ERRO 01: \n\nProblema de comunicação com o servidor.\n\nCaso o problema persista, favor entrar em contato com a equipe técnica.');   

        }       

    };

    /**
     * Método para fazer a tomada de decisão entre salvar o apagar log in da memória
     * @author Pedro Biasutti
     * @param values - valores do formulário (nomeUsuario e senha)
     * @param checked - status do checkbox
     */
    gerenciaLogIn = async (values, checked) => {

        const nomeUsuario = values.nomeUsuario;
        const senha = values.senha;

        if (checked) {

            await this._storeData('nomeUsuario', nomeUsuario);
            await this._storeData('senha', senha);

        } else {

            await this.apagaLogIn();

        }

    };

    /**
     * Método para apagar as variáveis do log in da memória
     * @author Pedro Biasutti
     */
    apagaLogIn = async () => {

        await this._removeData('nomeUsuario');
        await this._removeData('senha');

    };

    /**
     * Método para salvar variável na memória do celular
     * @author Pedro Biasutti
     * @param storage_Key - chave de acesso (nome a ser salvo)
     * @param value - valor da variável
     */
    _storeData = async (storage_Key, value) => {

        try {

            await AsyncStorage.setItem(storage_Key, value);

            console.log('NÃO DEU ERRO SALVA NA MEMÓRIA');

        } catch (error) {

            console.log('DEU ERRO SALVA NA MEMÓRIA');

        }

    };
      
    /**
     * Método para acessar variável salva na memória do celular
     * @author Pedro Biasutti
     * @param storage_Key - chave de acesso (nome salvo)
     */
    _retrieveData = async (storage_Key) => {

        try {

            const value = await AsyncStorage.getItem(storage_Key);

            if(value !== null) {

                console.log('NÃO DEU ERRO PEGA DA MEMÓRIA');

                return value;

            } else {

                console.log('DEU ERRO PEGA DA MEMÓRIA');
                console.log('VALOR NULO');

                return 'erro';

            }

        } catch (error) {

            console.log('DEU ERRO PEGA DA MEMÓRIA');

            return 'erro';

        }

    };
    
    /**
     * Método para remover variável salva na memória do celular
     * @author Pedro Biasutti
     * @param storage_Key - chave de acesso (nome salvo)
     */
    _removeData = async (storage_Key) => {

        try {

            await AsyncStorage.removeItem(storage_Key);

            console.log('NÃO DEU ERRO REMOVE DA MEMÓRIA');

        } catch (error) {

            console.log('DEU ERRO REMOVE DA MEMÓRIA');

        }

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

        return (

            <ScrollView>
                
                <View style = {{alignItems: 'center'}}>

                    <Image
                        style = {styles.logo}
                        source = {require('./../assets/logo01_sb.png')}
                        resizeMode = 'contain'
                    />

                </View>
                                
                <Formik

                    initialValues = {{
                        nomeUsuario: '',
                        senha: ''
                    }}

                    onSubmit = { async (values, actions) => {

                        let validation = 0;

                        validation = await this.validaForm(values);
                        actions.setSubmitting(false);

                        if (validation === 5) {

                            // fazer teste com formikprops.values para limpar o q foi digitado
                            this.props.navigation.navigate('Menu', {nomeUsuario: values.nomeUsuario});

                        } else {

                            // Garante que só é salvo credenciais válidas na memória
                            await this.apagaLogIn();

                        }

                    }}
                    // validateOnBlur= {false}
                    validateOnChange = {false}
                    validationSchema = {validationSchema}
                    >
                    {formikProps => (
                        <React.Fragment>

                            <View>

                                <View style = {styles.containerStyle}>
                                    <Text style = {styles.labelStyle}>Usuario</Text>
                                    <TextInput
                                        placeholder = 'usuario'
                                        style = {styles.inputStyle}
                                        onChangeText = {formikProps.handleChange('nomeUsuario')}
                                        onBlur = {formikProps.handleBlur('nomeUsuario')}
                                        autoCapitalize = { 'none' }
                                        onSubmitEditing = {() => { this.senha.focus() }}
                                        ref = {(ref) => { this.nomeUsuario = ref; }}
                                        returnKeyType = { "next" }
                                    />

                                </View>

                                {formikProps.errors.nomeUsuario &&
                                    <View>
                                        <Text style = {{ color: 'red', textAlign: 'center'}}>
                                            {formikProps.touched.nomeUsuario && formikProps.errors.nomeUsuario}
                                        </Text>
                                    </View>
                                }

                            </View>

                            <View>

                                <View style = {styles.containerStyle}>
                                    <Text style = {styles.labelStyle}>Senha</Text>
                                    <TextInput
                                        placeholder = 'senha123'
                                        style = {styles.inputStyle}
                                        onChangeText = {formikProps.handleChange('senha')}
                                        onBlur = {formikProps.handleBlur('senha')}
                                        secureTextEntry
                                        ref = {(ref) => { this.senha = ref; }}
                                        returnKeyType={ "next" }
                                    />
                                </View>

                                {formikProps.errors.senha &&
                                    <View>
                                        <Text style = {{ color: 'red', textAlign: 'center'}}>
                                            {formikProps.touched.senha && formikProps.errors.senha}
                                        </Text>
                                    </View>
                                }

                            </View>

                            <View style = {{marginHorizontal: 10, marginVertical: -10}}>

                                <CheckBox
                                    left
                                    title='Salvar Log in'
                                    checked={this.state.checked}
                                    onPress={() => {
                                        this.setState({checked: !this.state.checked});
                                        this.gerenciaLogIn(formikProps.values,!this.state.checked);
                                        }
                                    }
                                    textStyle = {{fontSize: 16, fontWeight: '600'}}
                                    containerStyle = {{backgroundColor: 'white', borderColor: 'white'}}
                                />
                                
                            </View>

                            <View style = {{alignItems: 'center'}}>

                                <Text style = {styles.hiperlink}
                                    onPress = {() => this.props.navigation.navigate('RecuperarSenha')}
                                >
                                    Esqueceu sua senha ?
                                </Text>

                                {formikProps.isSubmitting ? (

                                    <View style = {styles.activity}>

                                        <ActivityIndicator/>
                                        
                                    </View>

                                    ) : (

                                    <View style = {{flexDirection: 'column', flex: 1, width: '50%'}}>

                                        <TouchableOpacity 
                                            style = {styles.button}
                                            onPress={formikProps.handleSubmit}
                                        >
                                            <Text style = {styles.text}>Log In</Text>
                                        </TouchableOpacity>

                                    </View>
                                )}
                                
                            </View>
                                    
                        </React.Fragment>
                    )}
                    
                    </Formik>

                    <Text style = {[styles.hiperlink, {marginBottom: 0}]}
                        onPress = {() => this.props.navigation.navigate('Cadastro')}
                    >
                    Cadastro
                    </Text>

                    <Text style = {[styles.hiperlink, {marginBottom: 20}]}
                        onPress = {() => this.temosUso()}
                    >
                    Termos de uso
                    </Text>

                    <View>

                        <Modal
                            transparent = {false}
                            visible = {this.state.exibe_termos_de_uso}
                            onRequestClose = {() => {
                            console.log('Modal has been closed.');
                            }}
                        >

                            <View style={styles.modal_termos_container}>

                                <ScrollView>

                                    <Markdown rules={rules}>{this.state.texto_termoUso}</Markdown>

                                </ScrollView>

                                <TouchableOpacity
                                    onPress = {() => {
                                        this.setState({ exibe_termos_de_uso: false });
                                    }}
                                    style = {styles.button}
                                >

                                    <Text style = {styles.text}>Fechar</Text>

                                </TouchableOpacity>

                            </View>

                        </Modal>

                        {/* <Modal
                            transparent = {false}
                            visible = {this.state.exibe_instrucoes_de_uso}
                            onRequestClose = {() => {
                            console.log('Modal has been closed.');
                            }}
                        >

                            <View style={styles.modal_instrucoes_container}>

                                <Image 
                                    source = {require('./../assets/teste.gif')}
                                    style={styles.gif} 
                                />

                                <TouchableOpacity
                                    onPress = {() => {
                                        this.setState({ exibe_instrucoes_de_uso: false });
                                    }}
                                    style = {styles.button}
                                >

                                    <Text style = {styles.text}>Fechar</Text>

                                </TouchableOpacity>

                            </View>


                        </Modal> */}

                    </View>

            </ScrollView>

        );
        
    };
    
};

export default Home;

const styles = StyleSheet.create({
    
    button: {
        alignSelf: 'stretch',
        backgroundColor: '#39b500',
        borderRadius: 5,
        borderWidth: 1,
        borderColor: '#39b500',
        marginHorizontal: 5,
        marginVertical: 10,
    }, 
    text: {
        alignSelf: 'center',
        fontSize: 20,
        fontWeight: '600',
        color: 'white',
        paddingVertical: 10
    },
    inputStyle:{
        flex:2,
        fontSize: 18,
        lineHeight: 23,
        color: 'black',
        paddingHorizontal: 5,
    },
    labelStyle: {
        flex: 1,
        fontSize: 18,
        paddingLeft: 20,
    },
    containerStyle: {
        flex: 1,
        flexDirection: 'row',
        alignSelf: 'center',
        alignItems: 'center',
        backgroundColor: '#f2f2f2',
        height: 50,
        marginVertical: 5,
        marginHorizontal: 20
    },
    hiperlink: {
        alignSelf: 'center',
        fontSize: 18,
        textDecorationLine: 'underline', 
        color: '#39b500',
        marginVertical: 10 
    },
    logo: {
        width: 0.45 * screenWidth, 
        height: 0.25 * screenHeight
    },
    activity : {
        marginVertical: 20,
        transform: ([{ scaleX: 1.5 }, { scaleY: 1.5 }]),
    },
    modal_termos_container: {
        flex: 1,
        flexDirection: 'column',
        justifyContent: 'center',
        alignItems: 'center',
        marginVertical: 20,
        marginHorizontal: 10
    },
    modal_instrucoes_container: {
        flex: 1,
        alignItems: 'center',
        justifyContent: 'center',
        marginVertical: 20,
        marginHorizontal: 20,
    },
    gif: {
        width: 0.9 * screenWidth,
        height: 0.9 * screenWidth
    }


});

