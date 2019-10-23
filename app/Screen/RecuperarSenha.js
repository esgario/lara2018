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

const urlGetEmail = `${URL_API}/usuario/search/findByEmail`;
const urlPost = `${URL_API}-aberta/recuperar-senha`;

const validationSchema = yup.object().shape({

    senhaAntiga: yup
    .string()
    .required('Senha não foi informada')
    .min(4, 'Senha muito curta')
    .max(20, 'Senha muito longa'),
    senhaNova: yup
    .string()
    .required('Senha não foi informada')
    .min(4, 'Senha muito curta')
    .max(20, 'Senha muito longa'),
    confirmaSenha: yup
    .string()
    .required('Confirmação de senha não informada')
    .test('senhas-match', 'Senhas não conferem', function(value) {
        return this.parent.senhaNova === value;
    }),

});

const validationSchema2 = yup.object().shape({

    emailRecuperacao: yup
    .string()
    .required('Email não foi informado')
    .email()
    
});

class RecuperarSenha extends Component {

    state = {
        emailLogado: '',
        urlPatch: '',
        logStatus: false
    };

    static navigationOptions = {

        title: 'Recuperar Senha',
        headerStyle: {
            backgroundColor: '#39b500',
        },
        headerTintColor: 'white',
        headerTitleStyle: {
            fontWeight: 'bold',
            fontSize: 25
        },
    };

    /**
     * Método que retorna o usuário sendo passado seu email.
     * @author Pedro Biasutti
     * @param values - dados que vem do form.
     * @param actions - usado para setar submit false
     */
    getUserByEmail = async (values,actions) => {

        let urlPatch = '';
        let validation = 0;

        await axios({
            method: 'get',
            url: urlGetEmail,
            params: {
                email: values.emailRecuperacao,
            },
            headers: { 
                'Cache-Control': 'no-store',
            }
        })
        .then (function(response) {
            console.log('NÃO DEU ERRO PEGA USUARIO POR EMAIL')
            urlPatch = response.data._links.self.href;
            validation = 7;

        })
        .catch (function(error) {
            console.log('DEU ERRO PEGA USUARIO POR EMAIL')
        })

        if ( validation === 7 ) {

            this.setState({
                emailLogado: values.emailRecuperacao,
                urlPatch: urlPatch,
            });

            this.alteraSenha(values.emailRecuperacao,actions);

        } else {

            alert('O email não existe no banco de dados.\n\n Favor alterar este campo de dados !');

        }

    };

    /**
     * Método para gerar uma nova senha aleatória e envia-la por email, dado o email informado no campo.
     * @author Pedro Biasutti
     * @param email - email informado para recuperação de senha
     * @param actions - usado para setar submit false
     */
    alteraSenha = async (email,actions) => {

        let httpStatus = 0;

        await axios({
            method: 'post',
            url: urlPost,
            data: {
                email: email,
            },
            headers: { 
                'Cache-Control': 'no-store',
            }
        })
        .then (function(response) {
            console.log('NÃO DEU ERRO ALTERA SENHA');
            httpStatus = response.status;

        })
        .catch (function(error){
            console.log('DEU ERRO ALTERA SENHA');
            console.warn(error.request.status);
            console.warn(error);            
        })

        actions.setSubmitting(false);

        if ( httpStatus === 200 ) {

            var texto = 'Recuperação de senha realizada com sucesso. Acesse o email informado para mais detalhes.\n\n' +
            'Ao pressionar "Finalizar" o App o direcionará para pagina principal\n\n' +
            'Caso queira alterar a senha gerada por uma de sua preferência, clique em "Alterar".'

                Alert.alert(
                    'Atenção',
                    texto,
                    [             
                        {text: 'Finalizar', onPress: () => {
                            this.setState({logStatus: false});
                            this.props.navigation.navigate('Home');
                            }
                        },
                        {text: 'Alterar', onPress: () => this.setState({logStatus: true})}
                    ],
                    { cancelable: false }
                )         
        }
        
    };

    /**
     * Método para atualizar a senha do usuário dado a senha gerada/nova
     * @param email - email informado para recuperação de senha
     * @param actions - usado para setar submit false
     */

    atualizaSenha = async (values, actions) => {

        let httpStatus = 0;

        await axios({
            method: 'patch',
            url: this.state.urlPatch,
            data: {
                senha: values.senhaNova,
            },
            headers: { 
                'Cache-Control': 'no-store',
            }
        })
        .then (function(response) {
            // console.warn(response.status)
            console.log('NÃO DEU ERRO ATUALIZA SENHA');
            httpStatus = response.status;
        })
        .catch (function(error){
            // console.warn(error);
            console.log('DEU ERRO ATUALIZA SENHA');
            httpStatus = error.request.status;
        })

        actions.setSubmitting(false);

        if ( httpStatus === 200) {
    
            var texto = 'Alteração de senha realizada com sucesso.\n\n' +
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

    };

    render () {

        return (
            <ScrollView>
                <SafeAreaView style = {{ marginTop: 20}}>

                    {!this.state.logStatus && 

                        <View style = {{alignItems: 'center', justifyContent: 'center', marginBottom:30}}>
                            <Text style = {styles.headers}>Email de recuperação</Text>
                        </View>

                    }

                    {this.state.logStatus && 

                        <View style = {{alignItems: 'center', justifyContent: 'center', marginBottom:30}}>
                            <Text style={styles.headers}>Altere sua senha</Text>
                        </View>

                    }

                    { !this.state.logStatus &&
                        <Formik
                            initialValues = {{
                                emailUser: '',
                            }}
                            
                            onSubmit = { async (values, actions) => {

                                await this.getUserByEmail(values, actions);

                            }}
                            // validateOnBlur = {false}
                            validateOnChange = {false}
                            validationSchema = {validationSchema2}
                        >
                            {formikProps => (
                                <React.Fragment>
                                
                                <View>
    
                                    <View style = {styles.containerStyle}>
                                        <Text style = {styles.labelStyle}>Email</Text>
                                        <TextInput
                                            placeholder = 'john.snow@got.com'
                                            style = {styles.inputStyle}
                                            onChangeText = {formikProps.handleChange('emailRecuperacao')}
                                            onBlur = {formikProps.handleBlur('emailRecuperacao')}
                                        />

                                    </View>

                                    {formikProps.errors.senhaAntiga &&
                                        <View>
                                            <Text style = {{ color: 'red', textAlign: 'center'}}>
                                                {formikProps.touched.senhaAntiga && formikProps.errors.senhaAntiga}
                                            </Text>
                                        </View>
                                    }

                                </View>

                            
                                    <View style = {{alignItems: 'center', marginTop: 30}}>
                                        {formikProps.isSubmitting ? (
                                            <ActivityIndicator/>
                                            ) : (
                                            <View style = {{flexDirection: 'column', flex: 1, width: '50%'}}>

                                                <TouchableOpacity 
                                                    style = {styles.button}
                                                    onPress={formikProps.handleSubmit}
                                                >
                                                    <Text style = {styles.text}>Confirmar</Text>
                                                </TouchableOpacity>
    
                                            </View>
                                        )}
                                    </View>
                                            
                                </React.Fragment>
                            )}
                        </Formik>
                    }

                    { this.state.logStatus &&
                         <Formik
                            initialValues={{
                                senhaAntiga: '',
                                senhaNova: '',
                                confirmaSenha: '',
                            }}
                            onSubmit =  { async (values, actions) => {
                                
                                await this.atualizaSenha(values, actions);
                                
                            }}
                            // validateOnBlur = {false}
                            validateOnChange = {false}
                            validationSchema = {validationSchema}
                        >
                            {formikProps => (
                                <React.Fragment>
    
                                    <View>

                                        <View style={styles.containerStyle}>
                                            <Text style={styles.labelStyle}>Senha{'\n'}antiga</Text>
                                            <TextInput
                                                placeholder = 'senha123'
                                                style = {styles.inputStyle}
                                                onChangeText = {formikProps.handleChange('senhaAntiga')}
                                                onBlur = {formikProps.handleBlur('senhaAntiga')}
                                                secureTextEntry
                                                onSubmitEditing = {() => { this.senhaNova.focus() }}
                                                ref = {(ref) => { this.senhaAntiga = ref; }}
                                                returnKeyType = { "next" }
                                            />

                                        </View>

                                        {formikProps.errors.senhaAntiga &&
                                            <View>
                                                <Text style={{ color: 'red', textAlign: 'center'}}>
                                                    {formikProps.touched.senhaAntiga && formikProps.errors.senhaAntiga}
                                                </Text>
                                            </View>
                                        }

                                    </View>
    
                                    <View>
    
                                        <View style = {styles.containerStyle}>
                                            <Text style = {styles.labelStyle}>Senha</Text>
                                            <TextInput
                                                placeholder = 'senha456'
                                                style = {styles.inputStyle}
                                                onChangeText = {formikProps.handleChange('senhaNova')}
                                                onBlur = {formikProps.handleBlur('senhaNova')}
                                                secureTextEntry
                                                onSubmitEditing = {() => { this.confirmaSenha.focus() }}
                                                ref = {(ref) => { this.senhaNova = ref; }}
                                                returnKeyType={ "next" }
                                            />
    
                                        </View>
    
                                        {formikProps.errors.senhaNova &&
                                            <View>
                                                <Text style = {{ color: 'red', textAlign: 'center'}}>
                                                    {formikProps.touched.senhaNova && formikProps.errors.senhaNova}
                                                </Text>
                                            </View>
                                        }
    
                                    </View>
    
                                    <View>
    
                                        <View style = {styles.containerStyle}>
                                            <Text style = {styles.labelStyle}>Repita{'\n'}a senha</Text>
                                            <TextInput
                                                placeholder = 'senha456'
                                                style = {styles.inputStyle}
                                                onChangeText = {formikProps.handleChange('confirmaSenha')}
                                                onBlur = {formikProps.handleBlur('confirmaSenha')}
                                                secureTextEntry
                                                ref = {(ref) => { this.confirmaSenha = ref; }}
                                            />  
    
                                        </View>
    
                                        {formikProps.errors.confirmaSenha &&
                                            <View>
                                                <Text style = {{ color: 'red', textAlign: 'center'}}>
                                                    {formikProps.touched.confirmaSenha && formikProps.errors.confirmaSenha}
                                                </Text>
                                            </View>
                                        }
    
                                    </View>
    
                                    
    
                                    <View style = {{alignItems: 'center', marginTop: 30}}>
                                        {formikProps.isSubmitting ? (
                                            <ActivityIndicator/>
                                            ) : (
                                            <View style = {{flexDirection: 'column', flex: 1, width: '50%'}}>
    
                                                <TouchableOpacity 
                                                    style = {styles.button}
                                                    onPress={formikProps.handleSubmit}
                                                >
                                                    <Text style = {styles.text}>Alterar</Text>
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

export default RecuperarSenha;

const fontStyle = Platform.OS === 'ios' ? 'Arial Hebrew' : 'serif';

const styles = StyleSheet.create({
    headers: { 
        alignItems: 'center',
        fontSize: 28,
        fontWeight: 'bold',
        fontFamily: fontStyle,
        color: '#39b500',
        marginTop: 20,
    },
    button: {
        alignSelf: 'stretch',
        backgroundColor: '#39b500',
        borderRadius: 5,
        borderWidth: 1,
        borderColor: '#39b500',
        marginHorizontal: 5,
        marginVertical: 20
    }, 
    text: {
        alignSelf: 'center',
        color: 'white',
        fontSize: 16,
        fontWeight: '600',
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
        fontSize: 18,
        paddingLeft: 20,
        flex: 1,
    },
    containerStyle: {
        flex: 1,
        flexDirection: 'row',
        alignSelf: 'center',
        alignItems: 'center',
        height: 50,
        // borderWidth: 1,
        // borderRadius: 4,
        // borderColor: 'black',
        backgroundColor: '#f2f2f2',
        marginBottom: 10,
        marginHorizontal: 10,        

    }
});