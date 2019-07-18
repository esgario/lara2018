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
    TouchableOpacity
  } from 'react-native';

import { Formik } from 'formik';
import * as yup from 'yup';
import { ScrollView } from 'react-native-gesture-handler';
import axios from 'axios';
import { URL_API } from '../Utils/url_api';

// const urlGetNomeCompleto = `http://192.168.0.160:8080/api/usuario/search/findByNomeCompleto`;
// const urlGetNomeUsuario = `http://192.168.0.160:8080/api/usuario/search/findByNomeUsuario`;
// const urlGetEmail = `http://192.168.0.160:8080/api/usuario/search/findByEmail`;

const urlGetNomeCompleto = `${URL_API}/usuario/search/findByNomeCompleto`;
const urlGetNomeUsuario = `${URL_API}/usuario/search/findByNomeUsuario`;
const urlGetEmail = `${URL_API}/usuario/search/findByEmail`;

const urlPost = `${URL_API}/usuario`;

const FieldWrapper = ({ children, label, formikProps, formikKey }) => (

    <View style={{ marginHorizontal: 20, marginVertical: 10 }}>
        <Text style={{ marginBottom: 10, fontSize:18 }}>{label}</Text>
        {children}
        <Text style={{ color: 'red' }}>
            {formikProps.touched[formikKey] && formikProps.errors[formikKey]}
        </Text>
    </View>

);

const StyledSwitch = ({ formikKey, formikProps, label, ...rest }) => (
    <FieldWrapper label={label} formikKey={formikKey} formikProps={formikProps}>
        <Switch
            value={formikProps.values[formikKey]}
            style= {styles.switch}
            onValueChange={value => {
              formikProps.setFieldValue(formikKey, value);
            }}
            {...rest}
        />
    </FieldWrapper>
);

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
    .string()
    .required('Estado não informado'),
    cidade: yup
    .string()
    .required('Cidade não informado'),
    senha: yup
    .string()
    .required('Senha não foi informada')
    .min(4, 'Senha muito curta')
    .max(20, 'Senha muito longa'),
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
})

class Cadastro extends Component {

    static navigationOptions = {
        title: 'Cadastro',
        headerStyle: {
          backgroundColor: '#39b500',
        },
        headerTintColor: 'white',
        headerTitleStyle: {
          fontWeight: 'bold',
          fontSize: 25
        },
    };

    state = {
        validationG: 0,
        httpStatusG: 0,
    };

    constructor(props) {
        super(props);
    
        this.focusNextField = this.focusNextField.bind(this);
        this.inputs = {};
    }

    focusNextField(id) {
        this.inputs[id].focus();
    }

    /**
     * Método para checar se os campos 'nomeCompleto', 'nomeUsuario' e 'email' já foram salvos anteriormente no banco.
     * @author Pedro Biasutti
     * @param values - Dados que foram digitados no form.
     */
    checaDuplicidade = async (values) => {
        
        let validation = 0;

        // console.log('values.nomeCompleto', values.nomeCompleto);
        // console.log('values.nomeUsuario', values.nomeUsuario);
        // console.log('values.email', values.email);

        //  Verificar se o nomeCompleto digitado já existe no banco de dados
        await axios({
            method: 'get',
            url: urlGetNomeCompleto,
            params: {
                nomeCompleto: values.nomeCompleto,
            }
        })
        .then (function(response) {
            validation = validation + 1;
        })
        .catch (function(error){
            // console.warn(error);
        })

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

            alert('O nome completo já existe no banco de dados.\n\n Favor alterar este campo de dados !');

        } else if ( validation === 3) {

            alert('O nome do usuário já existe no banco de dados.\n\n Favor alterar este campo de dados !');

        } else if ( validation === 5) {

            alert('O email já existe no banco de dados.\n\n Favor alterar este campo de dados !');

        } 
        else if ( validation === 4) {

            alert('O nome completo e o nome do usuário já existem no banco de dados.\n\n Favor alterar estes campos de dados !');

        } 
        else if ( validation === 6) {

            alert('O nome completo e o email já existem no banco de dados.\n\n Favor alterar estes campos de dados !');

        } else if ( validation === 8) {

            alert('O nome do usuário e o email já existem no banco de dados.\n\n Favor alterar estes campos de dados !');

        } else if ( validation === 9) {

            alert('O nome completo, o nome do usuário e o email já existem no banco de dados.\n\n Favor alterar estes campos de dados !');

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

        await axios({
            method: 'post',
            url: urlPost,
            data: {
                nomeCompleto: values.nomeCompleto,
                nomeUsuario: values.nomeUsuario,
                email: values.email,
                estado: values.estado,
                cidade: values.cidade,
                senha: values.senha,
                apto: 'true',
                papel: 'USER'
            }
        })
        .then (function(response) {
            console.log('NÃO DEU ERRO CADASTRO USUÁRIO');
            // console.warn(response.status);
            // console.log('Http status: ',response.status);
            httpStatus = response.status;
        })
        .catch (function(error){
            console.log('DEU ERRO CADASTRO USUÁRIO');
            // console.warn(error.request.status);
            if (httpStatus !== 201) {
                httpStatus = error.request.status;
            }
        })

        this.setState({httpStatusG: httpStatus})
        
        if (this.state.httpStatusG === 201) {

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
            )
        
        }

    };

    render () {
        return (
            <ScrollView>
                <SafeAreaView style={{ justifyContent: 'space-around'}}>

                    <Formik
                        initialValues={{
                            nomeCompleto: '',
                            nomeUsuario: '',
                            email: '',
                            estado: '',
                            cidade: '',
                            senha: '',
                            confirmaSenha: '',
                            aceitaTermos: false
                        }}
                        onSubmit =  { async (values, actions) => {

                            await this.checaDuplicidade(values);
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

                                    <View style={{alignItems: 'center', justifyContent: 'center', marginBottom: 20}}>
                                        <Text style={styles.headers}>Crei sua conta</Text>
                                    </View>

                                    <View style={styles.containerStyle}>
                                        <Text style={styles.labelStyle}>Nome</Text>
                                        <TextInput
                                            placeholder='John Snow'
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
                                            placeholder='johnsnow'
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
                                            placeholder='john.snow@got.com'
                                            style={styles.inputStyle}
                                            onChangeText={formikProps.handleChange('email')}
                                            onBlur={formikProps.handleBlur('email')}
                                            onSubmitEditing={() => { this.estado.focus() }}
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
                                        <Text style={styles.labelStyle}>Estado</Text>
                                        <TextInput
                                            placeholder='Espírito Santo'
                                            style={styles.inputStyle}
                                            onChangeText={formikProps.handleChange('estado')}
                                            onBlur={formikProps.handleBlur('estado')}
                                            onSubmitEditing={() => { this.cidade.focus() }}
                                            ref={(ref) => { this.estado = ref; }}
                                            returnKeyType={ "next" }
                                        />

                                    </View>

                                    {formikProps.errors.estado &&
                                        <View>
                                            <Text style={{ color: 'red', textAlign: 'center'}}>
                                                {formikProps.touched.estado && formikProps.errors.estado}
                                            </Text>
                                        </View>
                                    }

                                </View>

                                <View>

                                    <View style={styles.containerStyle}>
                                        <Text style={styles.labelStyle}>Cidade</Text>
                                        <TextInput
                                            placeholder='Vitória'
                                            style={styles.inputStyle}
                                            onChangeText={formikProps.handleChange('cidade')}
                                            onBlur={formikProps.handleBlur('cidade')}
                                            onSubmitEditing={() => { this.senha.focus() }}
                                            ref={(ref) => { this.cidade = ref; }}
                                            returnKeyType={ "next" }
                                        />

                                    </View>

                                    {formikProps.errors.cidade &&
                                        <View>
                                            <Text style={{ color: 'red', textAlign: 'center'}}>
                                                {formikProps.touched.cidade && formikProps.errors.cidade}
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

                                

                                <StyledSwitch
                                    label='Aceita os termos'
                                    formikKey='aceitaTermos'
                                    formikProps={formikProps}
                                /> 

                                <View style={{alignItems: 'center'}}>
                                    {formikProps.isSubmitting ? (
                                        <ActivityIndicator/>
                                        ) : (
                                            <View style={{flexDirection: 'column', flex: 1, width: '50%'}}>

                                            <TouchableOpacity 
                                                style={styles.button}
                                                onPress={formikProps.handleSubmit}
                                            >
                                                <Text style={styles.text}>Sign Up</Text>
                                            </TouchableOpacity>
                        
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
}

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
    switch: {
        transform: switchScale,
        alignSelf: 'baseline',
        marginLeft: 20, 
    }
});
