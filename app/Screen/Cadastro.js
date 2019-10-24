import React, { Component } from 'react';

import {
    SafeAreaView,
    TextInput,
    ActivityIndicator,
    Text,
    View,
    Platform,
    StyleSheet,
    Switch,
    Alert,
    ScrollView,
    Modal,
    TouchableOpacity
  } from 'react-native';

import { Formik } from 'formik';
import axios from 'axios';
import * as yup from 'yup';

import * as Location from 'expo-location';
import * as Permissions from 'expo-permissions';

import Markdown from 'react-native-markdown-renderer';

import StyledButton from '../Style/Button';

import { URL_API } from '../Utils/url_api';

// Http request
const urlValidaNovosDados = `${URL_API}/usuario/validaNovosDados`;
const urlCadastraUsuario = `${URL_API}/usuario`;
const urlPegaLocalizacao = `${URL_API}/python/pegaLocalizacao`
const urlGetTermosUso = `${URL_API}/termos/search/findByTitulo`;

// Formik wrapper
const FieldWrapper = ({ children, label, formikProps, formikKey }) => (

    <View style = {{ marginLeft: 20, marginRight: 5, marginVertical: 5 }}>
        <Text style = {{ marginBottom: 5, fontSize:20 }}>{label}</Text>
        {children}
        <Text style = {{ color: 'red' }}>
            {formikProps.touched[formikKey] && formikProps.errors[formikKey]}
        </Text>
    </View>

);

// Formik Switch button
const StyledSwitch = ({ formikKey, formikProps, label, ...rest }) => (

    <FieldWrapper label={label} formikKey={formikKey} formikProps={formikProps}>
        <Switch
            value={formikProps.values[formikKey]}
            style= {{ transform: switchScale, alignSelf: 'baseline' }}
            onValueChange={value => {
              formikProps.setFieldValue(formikKey, value);
            }}
            {...rest}
        />
    </FieldWrapper>

);

// YUP validation
const validationSchema = yup.object().shape({

    nomeCompleto: yup
    .string()
    .required('Nome completo não foi informado'),
    nomeUsuario: yup
    .string()
    .required('Nome do usuário não foi informado'),
    email: yup
    .string()
    .required('Email não foi informado')
    .email('Email não é valido'),
    estado: yup
    .string(),
    cidade: yup
    .string(),
    senha: yup
    .string()
    .required('Senha não foi informada')
    .min(4, 'Senha muito curta')
    .max(10, 'Senha muito longa'),
    confirmaSenha: yup
    .string()
    .required('Confirmação de senha não informada')
    .test('senhas-match', 'Senhas não conferem', function(value) {
        return this.parent.senha === value;
    }),
    aceitaTermos: yup
    .boolean()
    .test(
        'is-true',
        'Aceite os termos para continuar',
        value => value === true
    )

});

class Cadastro extends Component {

    static navigationOptions = {

        title: 'Cadastro',
        headerStyle: {
            backgroundColor: '#39b500',
        },
        headerTintColor: 'white',
        headerTitleStyle: {
        fontWeight: 'bold',
        fontSize: 26
        }

    };

    state = {

        validationG: 0,
        httpStatusG: 0,
        cidade: '',
        estado: '',
        hasLocationPermission: null,
        exibeModal: false,
        texto_termoUso: null

    };

    constructor(props) {

        super(props);
        this.focusNextField = this.focusNextField.bind(this);
        this.inputs = {};

    };

    /**
     * Método chamado ao montar a tela
     * @author Pedro Biasutti
     */
    componentDidMount () {        

        // Pega as coordenadas e e tiver as coordenadas, pega a localização
        this.pegaCoordenadas();

    };

    /**
     * Método para pegar a localização do usuário e salva-la
     * @author Pedro Biasutti
     */
    pegaCoordenadas = async () => {

        const { status } = await Permissions.askAsync(Permissions.LOCATION);
        this.setState({ hasLocationPermission: status === 'granted' });

        const permission = await Permissions.getAsync(Permissions.LOCATION);

        if (permission.status !== 'granted') {

            this.setState({latitude: null, longitude: null})

            console.log('DEU ERRO PEGA COORDENADAS');

        }   else {

            let location = await Location.getCurrentPositionAsync({});

            console.log(location);

            let coord = location.coords.latitude + ', ' + location.coords.longitude;

            this.pegaLocalizacao(coord);

        }     

    };

    /**
     * Método que retorna a cidade e estado dado as coordenadas "latitude, longitude"
     * @author Pedro Biasutti
     * @param coord - coordenadas
     */
    pegaLocalizacao = async (coord) => {

        let resp = '';
        let estado = '';
        let cidade = '';

        await axios({
            method: 'get',
            url: urlPegaLocalizacao,
            params: {
                coord: coord,
            },
            headers: { 
                'Cache-Control': 'no-store',
            }
        })
        .then (function(response) {
            resp = response.data;
            console.log(resp);
        })
        .catch (function(error){
            console.log('DEU ERRO PEGA LOCALIZACAO');
            console.log(error);
        })

        if (resp !== '' && resp !== 'error'){

            // Pega o estado e cidade, caso não seja None (nulo)
            resp.split('\n')[0] === 'None' ? estado = '' : estado = resp.split('\n')[0];
            resp.split('\n')[1] === 'None' ? cidade = '' : cidade = resp.split('\n')[1];

            // Salva as variaveis globalmente
            this.setState({estado, cidade})

            console.log('NÃO DEU ERRO PEGA LOCALIZACAO');

        } else {

            console.log('DEU ERRO PEGA LOCALIZACAO');

        }

    };

    /**
     * Método para trocar o foco para o proximo campo a ser digitado
     * @author Pedro Biasutti
     * @param {*} id 
     */
    focusNextField(id) {

        this.inputs[id].focus();

    };

    /**
     * Método para checar se os campos 'nomeCompleto', 'nomeUsuario' e 'email' já foram salvos anteriormente no banco.
     * @author Pedro Biasutti
     * @param values - Dados que foram digitados no form.
     */
    validaNovosDados = async (values) => {

        let validation = 0;

        await axios({
            method: 'get',
            url: urlValidaNovosDados,
            params: {
                dados: values,
                nomeUsuarioAntigo: this.state.nomeUsuarioLogado
            },
            headers: { 
                'Cache-Control': 'no-store',
            }
        })
        .then (function(response) {
            console.log('NÃO DEU ERRO VALIDA NOVOS DADOS');
            validation = response.data;
        })
        .catch (function(error){
            console.log('DEU ERRO VALIDA NOVOS DADOS');
        })

        this.setState({validationG: validation});

    };

    /**
     * Método para verificar se existe duplicidade no banco de dados com o form digitado.
     * Caso afirmativo, o post não é realizado. Caso negativo, o post é feito.
     * @author Pedro Biasutti
     * @param values - Dados que foram digitados no form.
     */
    validaForm = (values) => {

        let validation = this.state.validationG; //serve como parâmetro para a realização ou não do post.

        if ( validation === 1) {

            this.geraAlerta('O nome completo já existe no banco de dados.\n\n Favor alterar este campo de dados !');

        } else if ( validation === 3) {

            this.geraAlerta('O nome do usuário já existe no banco de dados.\n\n Favor alterar este campo de dados !');

        } else if ( validation === 5) {

            this.geraAlerta('O email já existe no banco de dados.\n\n Favor alterar este campo de dados !');

        } 
        else if ( validation === 4) {

            this.geraAlerta('O nome completo e o nome do usuário já existem no banco de dados.\n\n Favor alterar estes campos de dados !');

        } 
        else if ( validation === 6) {

            this.geraAlerta('O nome completo e o email já existem no banco de dados.\n\n Favor alterar estes campos de dados !');

        } else if ( validation === 8) {

            this.geraAlerta('O nome do usuário e o email já existem no banco de dados.\n\n Favor alterar estes campos de dados !');

        } else if ( validation === 9) {

            this.geraAlerta('O nome completo, o nome do usuário e o email já existem no banco de dados.\n\n Favor alterar estes campos de dados !');

        } else {

            this.postForm(values);
                
        }    

    };

    /**
     * Método para realizar o post do form. A necessidade de criar uma função veio do fato de que expressões condicionais não aceitam ser assíncronas.
     * @author Pedro Biasutti
     * @param values - Dados que foram digitados no form.
     */
    postForm = async (values) => {

        this.setState({validationG: 0});

        let httpStatus = 0; // indicará a resposta da requisição http.

        // Checa validade dos campos Estado e Cidade
        values = await this.checaEstadoCidade(values);

        await axios({
            method: 'post',
            url: urlCadastraUsuario,
            headers: {
                'Content-Type': 'application/json',
                'Cache-Control': 'no-store',
            },
            data: {
                nomeCompleto: values.nomeCompleto,
                nomeUsuario: values.nomeUsuario,
                email: values.email,
                estado: values.estado,
                cidade: values.cidade,
                senha: values.senha,
                apto: true,
                papel: 'USER'
            }
        })
        .then (function(response) {
            console.log('NÃO DEU ERRO SING UP');
            httpStatus = response.status;
            console.log(response.data);
        })
        .catch (function(error){
            console.log('DEU ERRO SING UP');
        })
        
        if (httpStatus === 201) {

            var texto = 'Conta criada com sucesso.\n\n' +
            'Ao pressionar "ok" o App o direcionará para pagina principal\n\n'

            Alert.alert(
                'Atenção',
                texto,
                [             
                    {text: 'Ok', onPress: () => {
                        this.props.navigation.navigate('Home');
                        }
                    },
                ],
                { cancelable: false }
            );
        
        } else {

            this.geraAlerta('ERRO 01: \n\nProblema de comunicação com o servidor.\n\nCaso o problema persista, favor entrar em contato com a equipe técnica.');   

        }

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
                'Cache-Control': 'no-store',
            }
        })
        .then (function(response) {
            console.log('NÃO DEU ERRO PEGA TERMOS DE USO');
            texto = response.data.texto;

        })
        .catch (function(error){
            console.log('DEU ERRO PEGA TERMOS DE USO');       
        })

        this.setState({texto_termoUso: texto, exibeModal: true});

    };

    /**
     * Método para checar se Estado e Cidade foram ou não digitados.
     * Caso não foram, subtitui-los pelas posições do GPS
     * @author Pedro Biasutti
     * @param values - valores do form
     */
    checaEstadoCidade = (values) => {

        if (values.estado === '' && this.state.estado !== '') {

            values.estado = this.state.estado

        }

        if (values.cidade === '' && this.state.cidade !== '') {

            values.cidade = this.state.cidade

        }

        return values;

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

                <SafeAreaView style = {{ marginTop: 20}}>

                    <Formik
                        initialValues={{
                            nomeCompleto: '',
                            nomeUsuario: '',
                            email: '',
                            senha: '',
                            estado: this.state.estado,
                            cidade: this.state.cidade,
                            confirmaSenha: '',
                            aceitaTermos: false
                        }}
                        onSubmit =  { async (values, actions) => {

                            await this.validaNovosDados(values);
                            await this.validaForm(values);
                            
                            actions.setSubmitting(false);
                            
                            
                        }}
                        // validateOnBlur= {false}
                        validateOnChange={false}
                        validationSchema={validationSchema}
                    >
                        {formikProps => (
                            <React.Fragment>

                            <View>

                                <View style = {{alignItems: 'center', justifyContent: 'center', marginBottom: 20}}>

                                    <Text style = {styles.headers}>Crei sua conta</Text>
                                    
                                </View>

                                <View style = {styles.containerStyle}>

                                    <Text style = {styles.labelStyle}>Nome</Text>
                                    <TextInput
                                        placeholder = 'John Snow'
                                        style = {styles.inputStyle}
                                        onChangeText = {formikProps.handleChange('nomeCompleto')}
                                        onBlur = {formikProps.handleBlur('nomeCompleto')}
                                        onSubmitEditing = {() => { this.nomeUsuario.focus() }}
                                        ref = {(ref) => { this.nomeCompleto = ref; }}
                                        returnKeyType = { "next" }
                                    />

                                </View>

                                {formikProps.errors.nomeCompleto &&
                                    <View>

                                        <Text style = {{ color: 'red', textAlign: 'center'}}>
                                            {formikProps.touched.nomeCompleto && formikProps.errors.nomeCompleto}
                                        </Text>

                                    </View>
                                }

                            </View>

                            <View>

                                <View style = {styles.containerStyle}>

                                    <Text style = {styles.labelStyle}>Usuario</Text>
                                    <TextInput
                                        placeholder = 'johnsnow'
                                        style = {styles.inputStyle}
                                        onChangeText = {formikProps.handleChange('nomeUsuario')}
                                        onBlur = {formikProps.handleBlur('nomeUsuario')}
                                        onSubmitEditing = {() => { this.email.focus() }}
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

                                    <Text style = {styles.labelStyle}>Email</Text>
                                    <TextInput
                                        placeholder = 'john.snow@got.com'
                                        style = {styles.inputStyle}
                                        onChangeText = {formikProps.handleChange('email')}
                                        onBlur = {formikProps.handleBlur('email')}
                                        onSubmitEditing = {() => { this.estado.focus() }}
                                        ref = {(ref) => { this.email = ref; }}
                                        returnKeyType = { "next" }
                                    />

                                </View>

                                {formikProps.errors.email &&
                                    <View>

                                        <Text style = {{ color: 'red', textAlign: 'center'}}>
                                            {formikProps.touched.email && formikProps.errors.email}
                                        </Text>

                                    </View>
                                }

                            </View>

                            <View>

                                <View style = {styles.containerStyle}>

                                    <Text style = {styles.labelStyle}>Estado</Text>
                                    <TextInput
                                        placeholder = {this.state.estado}
                                        style = {styles.inputStyle}
                                        onChangeText = {formikProps.handleChange('estado')}
                                        onBlur = {formikProps.handleBlur('estado')}
                                        onSubmitEditing = {() => { this.cidade.focus() }}
                                        ref = {(ref) => { this.estado = ref; }}
                                        returnKeyType = { "next" }
                                    />

                                </View>

                                {formikProps.errors.estado &&
                                    <View>

                                        <Text style = {{ color: 'red', textAlign: 'center'}}>
                                            {formikProps.touched.estado && formikProps.errors.estado}
                                        </Text>

                                    </View>
                                }

                            </View>

                            <View>

                                <View style = {styles.containerStyle}>

                                    <Text style = {styles.labelStyle}>Cidade</Text>
                                    <TextInput
                                        placeholder = {this.state.cidade}
                                        style = {styles.inputStyle}
                                        onChangeText = {formikProps.handleChange('cidade')}
                                        onBlur = {formikProps.handleBlur('cidade')}
                                        onSubmitEditing = {() => { this.senha.focus() }}
                                        ref = {(ref) => { this.cidade = ref; }}
                                        returnKeyType = { "next" }
                                    />

                                </View>

                                {formikProps.errors.cidade &&
                                    <View>

                                        <Text style = {{ color: 'red', textAlign: 'center'}}>
                                            {formikProps.touched.cidade && formikProps.errors.cidade}
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
                                        onSubmitEditing = {() => { this.confirmaSenha.focus() }}
                                        ref = {(ref) => { this.senha = ref; }}
                                        returnKeyType = { "next" }
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

                            <View>

                                <View style = {styles.containerStyle}>

                                    <Text style = {styles.labelStyle}>Repita{'\n'}a senha</Text>
                                    <TextInput
                                        placeholder = 'senha123'
                                        style = {styles.inputStyle}
                                        onChangeText = {formikProps.handleChange('confirmaSenha')}
                                        onBlur = {formikProps.handleBlur('confirmaSenha')}
                                        secureTextEntry
                                        ref = {(ref) => { this.confirmaSenha = ref; }}
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

                            
                            <View style = {{flexDirection: 'row'}}>
                                
                                <StyledSwitch
                                    label = 'Aceito os'
                                    formikKey = 'aceitaTermos'
                                    formikProps = {formikProps}
                                />

                                <Text style = {[styles.hiperlink]}
                                    onPress = {() => this.temosUso()}
                                >
                                termos de uso
                                </Text> 

                            </View>

                            <Modal
                                transparent = {false}
                                visible = {this.state.exibeModal}
                                onRequestClose = {() => {
                                console.log('Modal has been closed.');
                                }}
                            >

                                <View style={styles.modalContainer}>

                                    <ScrollView>

                                        <Markdown>{this.state.texto_termoUso}</Markdown>

                                    </ScrollView>

                                    <TouchableOpacity
                                        onPress = {() => {
                                            this.setState({ exibeModal: false });
                                        }}
                                        style = {styles.button}
                                    >

                                        <Text style = {styles.text}>Fechar</Text>

                                    </TouchableOpacity>

                                </View>


                            </Modal>

                            <View style = {{alignItems: 'center'}}>

                                {formikProps.isSubmitting ? (
                                    <View style = {styles.activity}>

                                        <ActivityIndicator/>

                                    </View>
                                    ) : (
                                        <View style = {styles.buttonContainer}>

                                            <StyledButton 
                                                onPress = {formikProps.handleSubmit}
                                            >
                                                Sign Up
                                            </StyledButton>
                    
                                    </View>
                                )}

                            </View>
                                    
                        </React.Fragment>

                        )}

                    </Formik>

                </SafeAreaView>

            </ScrollView>
            
        );
    }

};

export default Cadastro;

const fontStyle = Platform.OS === 'ios' ? 'Arial Hebrew' : 'serif';
const switchScale = Platform.OS === 'ios' ? [{ scaleX: 1.0 }, { scaleY: 1.0 }] : [{ scaleX: 1.3 }, { scaleY: 1.3 }];

const styles = StyleSheet.create({
    
    headers: { 
        fontFamily: fontStyle,
        color: '#39b500',
        fontWeight: 'bold',
        fontSize: 28,
        marginTop: 20,
        alignItems: 'center'
    },
    inputStyle:{
        color: 'black',
        paddingRight: 5,
        paddingLeft: 5,
        fontSize: 18,
        lineHeight: 23,
        flex:2,
    },
    labelStyle: {
        fontSize: 18,
        paddingLeft: 20,
        flex: 1,
    },
    containerStyle: {
        flex: 1,
        flexDirection: 'row',
        alignSelf: 'center',
        alignItems: 'center',
        backgroundColor: '#f2f2f2',
        height: 50,
        marginHorizontal: 20,
        marginBottom: 20,
    },
    switch: {
        transform: switchScale,
        alignSelf: 'baseline',
        marginLeft: 20, 
    },
    buttonContainer: {
        flex: 1, 
        flexDirection: 'column', 
        width: '50%',
        marginTop: '50%',
        marginBottom: '20%'
    },
    activity: {
        transform: ([{ scaleX: 1.5 }, { scaleY: 1.5 }]),
    },
    modalContainer: {
        flex: 1,
        flexDirection: 'column',
        justifyContent: 'center',
        alignItems: 'center',
        marginVertical: 20,
        marginHorizontal: 10
    },
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
    hiperlink: {
        alignSelf: 'flex-start',
        fontSize: 20,
        textDecorationLine: 'underline', 
        color: '#39b500',
        marginVertical: 5
    },

});
