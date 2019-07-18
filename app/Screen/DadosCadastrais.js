import React, { Component } from 'react';
import {
    SafeAreaView,
    TextInput,
    ActivityIndicator,
    Text,
    View,
    Platform,
    StyleSheet,
    Alert,
    TouchableOpacity
  } from 'react-native';

import { Formik } from 'formik';
import * as yup from 'yup';
import { ScrollView } from 'react-native-gesture-handler';
import axios from 'axios';
import { URL_API } from '../Utils/url_api';

// const urlGetNomeUsuario = `http://192.168.0.160:8080/api/usuario/search/findByNomeUsuario`;
// const urlGetEmail = `http://192.168.0.160:8080/api/usuario/search/findByEmail`;

const urlGetNomeUsuario = `${URL_API}/usuario/search/findByNomeUsuario`;
const urlGetEmail = `${URL_API}/usuario/search/findByEmail`;

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
    senha: yup
    .string()
    .required('Senha não foi informada')
    .min(2, 'Senha muito curta')
    .max(10, 'Senha muito longa'),
    confirmaSenha: yup
    .string()
    .required('Confirmação da senha não foi informada')
    .test('senhas-match', 'Senhas não conferem', function(value) {
        return this.parent.senha === value;
    }),
});

class RegistrationDataScreen extends Component {

    constructor (props) {
        super(props);
    };

    state = {
        nomeCompletoLogado: '',
        nomeUsuarioLogado: '',
        emailLogado: '',
        senhaLogado: '',
        urlPatch: '',
        logStatus: false
    };

    static navigationOptions = {

        title: 'Dados cadastrais',
        headerStyle: {
            backgroundColor: '#39b500',
        },
        headerTintColor: 'white',
        headerTitleStyle: {
            fontWeight: 'bold',
            fontSize: 25
        },
    };

    componentDidMount() {
        const { navigation } = this.props;
        const nomeUsuario = navigation.getParam('nomeUsuario', 'erro nomeUsuario');

        this.setState({nomeUsuarioLogado: nomeUsuario});

        console.log('nomeUsuario', nomeUsuario);
        console.log('this.state.nomeUsuarioLogado', this.state.nomeUsuarioLogado);

        this.getUserByNomeUsuario(nomeUsuario);
    };

    /**
     * Método que retorna o usuário sendo passado seu nome do usuário.
     * @author Pedro Biasutti
     * @param nomeUsuario - nome do usuário logado
     */
    getUserByNomeUsuario = async (nomeUsuario) => {

        console.log('this.state.nomeUsuarioLogado', this.state.nomeUsuarioLogado);

        let nomeUsuarioLogado = nomeUsuario;
        let urlPatch = '';
        let nomeCompleto = '';
        let senha = '';
        let validation = 0;

        await axios({
            method: 'get',
            url: urlGetNomeUsuario,
            params: {
                nomeUsuario: nomeUsuarioLogado,
            }
        })
        .then (function(response) {
            urlPatch = response.data._links.self.href;
            // console.warn(response.data);
            nomeCompleto = response.data.nomeCompleto;
            email = response.data.email;
            senha = response.data.senha;
            validation = 7;
            // console.log('nomeCompleto', nomeCompleto);
            // console.log('nomeUsuario', nomeUsuario);
        })
        .catch (function(error){
            console.warn(error);
        })

        if ( validation === 7 ) {

            this.setState({
                nomeCompletoLogado: nomeCompleto,
                senhaLogado: senha,
                emailLogado: email,
                urlPatch: urlPatch,
                logStatus: true
            });

        } else {

            alert('O nome do usuário não existe no banco de dados.\n\n Favor alterar este campo de dados !');

        }


    };

    /**
     * Método para checar se o'nomeUsuario'  e o 'email' já foram salvos anteriormente no banco, por outro usuário.
     * @author Pedro Biasutti
     * @param values - Dados que foram digitados no form.
     */
    checaDuplicidade = async (values) => {

        let validation = 0;

        // console.log('validation 0', validation);

        //  Verificar se o nomeUsuario digitado já existe no banco de dados
        await axios({
            method: 'get',
            url: urlGetNomeUsuario,
            params: {
                nomeUsuario: values.nomeUsuario,
            }
        })
        .then (function(response) {
            validation = validation + 3;
        })
        .catch (function(error){
        })

        // console.log('validation 1', validation);

        if ( values.nomeUsuario === this.state.nomeUsuarioLogado ) {
            validation = validation - 3;
        }

        // console.log('validation 2', validation);

        //  Verificar se o email digitado já existe no banco de dados
        await axios({
            method: 'get',
            url: urlGetEmail,
            params: {
                email: values.email,
            }
        })
        .then (function(response) {
            validation = validation + 5;
        })
        .catch (function(error){
        })

        // console.log('validation 3', validation);

        if ( values.email === this.state.emailLogado ) {

            validation = validation - 5;
        }

        // console.log('validation 4', validation);

        return validation;
    };

    /**
     * Método para verificar se existe duplicidade no banco de dados com o form digitado.
     * Caso afirmativo, o patch não é realizado. Caso negativo, o patch é feito.
     * @author Pedro Biasutti
     * @param validation - serve como parâmetro para a realização ou não do patch.
     * @param values - Dados que foram digitados no form.
     */
    validaForm = ( validation, values) => {

        // console.log('validation 01', validation);

        if ( validation === 3) {

            alert('O nome do usuário pentence a outro usuário.\n\n Favor alterar este campo de dados !');

        } else if (validation === 5) {

            alert('O email já existe no banco de dados.\n\n Favor alterar este campo de dados !');

        } else if ( validation === 8) {

            alert('O nome do usuário e o email já existem no banco de dados.\n\n Favor alterar estes campos de dados !');

        } else {

            this.setState({
                nomeCompletoLogado: values.nomeCompleto,
                nomeUsuarioLogado: values.nomeUsuario,
                senhaLogado: values.senha,
                emailLogado: values.email
            });
            
        }

    };

    /**
     * Método para atualizar algumas informações do usuário
     * @author Pedro Biasutti
     */
    atualizaUser = async () => {

        let urlPatch = this.state.urlPatch;
        let httpStatus = 0;

        await axios({
            method: 'patch',
            url: urlPatch,
            data: {
                nomeCompleto: this.state.nomeCompletoLogado,
                nomeUsuario: this.state.nomeUsuarioLogado,
                email: this.state.emailLogado,
                senha: this.state.senhaLogado
            }
        })
        .then (function(response) {
            // console.warn(response.status)
            console.log('NÃO DEU ERRO ATUALIZA USUÁRIO');
            httpStatus = response.status;
        })
        .catch (function(error){
            // console.warn(error);
            console.log('DEU ERRO ATUALIZA USUÁRIO');
            httpStatus = error.request.status;
        })

        // console.log('this.state.nomeCompletoLogado', this.state.nomeCompletoLogado);
        // console.log('this.state.nomeUsuarioLogado', this.state.nomeUsuarioLogado);
        // console.log('this.state.emailLogado', this.state.emailLogado);

        return httpStatus;

    };

    render () {

        return (
            <ScrollView>
                <SafeAreaView style={{ marginTop: 20}}>

                    <View style={{alignItems: 'center', justifyContent: 'center', marginBottom:30}}>
                        <Text style={styles.headers}>Alterar Dados Cadastrais</Text>
                    </View>

                    { this.state.logStatus &&
                        <Formik
                        initialValues={{
                            nomeCompleto: this.state.nomeCompletoLogado,
                            nomeUsuario: this.state.nomeUsuarioLogado,
                            email: this.state.emailLogado,
                            senha: '',
                            confirmaSenha: '',
                        }}
                        onSubmit =  { async (values, actions) => {
                            
                            let validation = 0;
                            let httpStatus = 0;

                            validation = await this.checaDuplicidade(values);
                            this.validaForm(validation, values);

                            // console.log('\n\nvalidation', validation);
                            // console.log(validation ==! 3 && validation ==! 5 && validation ==! 8);

                            if ( validation ==! 3 && validation ==! 5 && validation ==! 8) {
                                httpStatus = await this.atualizaUser();
                            }

                            if ( httpStatus === 200) {

                                var texto = 'Alteração cadastral realizada com sucesso.\n\n' +
                                            'Para retornar a pagina principal aperte "Ok".\n\nCaso queira permanecer nesta tela aperte "Cancelar".\n\n';

                                Alert.alert(
                                    'Atenção',
                                    texto,
                                    [             
                                        {text: 'Ok', onPress: () => {
                                            this.props.navigation.navigate('Home');
                                            }
                                        },
                                        {text: 'Cancelar'}
                                    ],
                                    { cancelable: false }
                                )

                            }

                            console.log('httpStatus', httpStatus);
                            actions.setSubmitting(false);



                        }}
                        // validateOnBlur= {false}
                        validateOnChange={false}
                        validationSchema={validationSchema}
                        >
                            {formikProps => (
                                <React.Fragment>

                                    <View>

                                        <View style={styles.containerStyle}>
                                            <Text style={styles.labelStyle}>Nome</Text>
                                            <TextInput
                                                placeholder={this.state.nomeCompletoLogado}
                                                style={styles.inputStyle}
                                                onChangeText={formikProps.handleChange('nomeCompleto')}
                                                onBlur={formikProps.handleBlur('nomeCompleto')}
                                                onSubmitEditing={() => { this.nomeUsuario.focus() }}
                                                ref={(ref) => { this.nomeCompleto = ref; }}
                                                returnKeyType={ "next" }
                                            />

                                        </View>

                                        {formikProps.errors.nomeCompleto &&
                                            <View>
                                                <Text style={{ color: 'red', textAlign: 'center'}}>
                                                    {formikProps.touched.nomeCompleto && formikProps.errors.nomeCompleto}
                                                </Text>
                                            </View>
                                        }

                                    </View>

                                    <View>

                                        <View style={styles.containerStyle}>
                                            <Text style={styles.labelStyle}>Usuario</Text>
                                            <TextInput
                                                placeholder={this.state.nomeUsuarioLogado}
                                                style={styles.inputStyle}
                                                onChangeText={formikProps.handleChange('nomeUsuario')}
                                                onBlur={formikProps.handleBlur('nomeUsuario')}
                                                onSubmitEditing={() => { this.email.focus() }}
                                                ref={(ref) => { this.nomeUsuario = ref; }}
                                                returnKeyType={ "next" }
                                            />

                                        </View>

                                        {formikProps.errors.nomeUsuario &&
                                            <View>
                                                <Text style={{ color: 'red', textAlign: 'center'}}>
                                                    {formikProps.touched.nomeUsuario && formikProps.errors.nomeUsuario}
                                                </Text>
                                            </View>
                                        }

                                    </View>

                                    <View>

                                        <View style={styles.containerStyle}>
                                            <Text style={styles.labelStyle}>Email</Text>
                                            <TextInput
                                                placeholder={this.state.emailLogado}
                                                style={styles.inputStyle}
                                                onChangeText={formikProps.handleChange('email')}
                                                onBlur={formikProps.handleBlur('email')}
                                                onSubmitEditing={() => { this.senha.focus() }}
                                                ref={(ref) => { this.email = ref; }}
                                                returnKeyType={ "next" }
                                            />

                                        </View>

                                        {formikProps.errors.email &&
                                            <View>
                                                <Text style={{ color: 'red', textAlign: 'center'}}>
                                                    {formikProps.touched.email && formikProps.errors.email}
                                                </Text>
                                            </View>
                                        }

                                    </View>

                                    <View>

                                        <View style={styles.containerStyle}>
                                            <Text style={styles.labelStyle}>Senha</Text>
                                            <TextInput
                                                placeholder='senha123'
                                                style={styles.inputStyle}
                                                onChangeText={formikProps.handleChange('senha')}
                                                onBlur={formikProps.handleBlur('senha')}
                                                secureTextEntry
                                                onSubmitEditing={() => { this.confirmaSenha.focus() }}
                                                ref={(ref) => { this.senha = ref; }}
                                                returnKeyType={ "next" }
                                            />

                                        </View>

                                        {formikProps.errors.senha &&
                                            <View>
                                                <Text style={{ color: 'red', textAlign: 'center'}}>
                                                    {formikProps.touched.senha && formikProps.errors.senha}
                                                </Text>
                                            </View>
                                        }

                                    </View>

                                    <View>

                                        <View style={styles.containerStyle}>
                                            <Text style={styles.labelStyle}>Repita{'\n'}a senha</Text>
                                            <TextInput
                                                placeholder='senha123'
                                                style={styles.inputStyle}
                                                onChangeText={formikProps.handleChange('confirmaSenha')}
                                                onBlur={formikProps.handleBlur('confirmaSenha')}
                                                secureTextEntry
                                                ref={(ref) => { this.confirmaSenha = ref; }}
                                            />

                                        </View>

                                        {formikProps.errors.senha &&
                                            <View>
                                                <Text style={{ color: 'red', textAlign: 'center'}}>
                                                    {formikProps.touched.senha && formikProps.errors.senha}
                                                </Text>
                                            </View>
                                        }

                                    </View>

                                    <View style={{alignItems: 'center', marginTop: 30, marginBottom: 30}}>
                                    {formikProps.isSubmitting ? (
                                        <ActivityIndicator/>
                                        ) : (
                                        <View style={{flexDirection: 'column', flex: 1, width: '50%'}}>

                                            <TouchableOpacity 
                                                style={styles.button}
                                                onPress={formikProps.handleSubmit}
                                            >
                                                <Text style={styles.text}>Alterar</Text>
                                            </TouchableOpacity>

                                        </View>
                                    )}
                                    </View>
                                            
                                </React.Fragment>
                            )}
                        </Formik>
                    }
                </SafeAreaView>
            </ScrollView>
            
            
        );
    }
}

export default RegistrationDataScreen;

const fontStyle = Platform.OS === 'ios' ? 'Arial Hebrew' : 'serif';

const styles = StyleSheet.create({
    headers: { 
        fontFamily: fontStyle,
        color: '#39b500',
        fontWeight: 'bold',
        fontSize: 28,
        marginTop: 20,
        alignItems: 'center'
    },
    button: {
        alignSelf: 'stretch',
        backgroundColor: '#39b500',
        borderRadius: 5,
        borderWidth: 1,
        borderColor: '#39b500',
        marginHorizontal: 5,
        marginBottom: 120
    }, 
    text: {
        alignSelf: 'center',
        color: 'white',
        fontSize: 16,
        fontWeight: '600',
        paddingVertical: 10,
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
        // borderWidth: 1,
        // borderRadius: 4,
        // borderColor: 'black',
    },
});
