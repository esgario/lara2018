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
    ScrollView
  } from 'react-native';

import StyledButton from '../Style/Button';
import { URL_API } from '../Utils/url_api';
import { Formik } from 'formik';
import * as yup from 'yup';
import axios from 'axios';

const urlGetNomeUsuario = `${URL_API}/usuario/search/findByNomeUsuario`;
const urlValidaAlteraDados = `${URL_API}/usuario/validaAlteraDados`;

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
    .required('Estado não foi informado'),
    cidade: yup
    .string()
    .required('Cidade não foi informada'),
    senhaAntiga: yup
    .string()
    .required('Senha antiga não foi informada')
    .min(2, 'Senha muito curta')
    .max(10, 'Senha muito longa'),
    senha: yup
    .string()
    .required('Senha não foi informada')
    .min(4, 'Senha muito curta')
    .max(10, 'Senha muito longa'),
    confirmaSenha: yup
    .string()
    .required('Confirmação da senha não foi informada')
    .test('senhas-match', 'Senhas não conferem', function(value) {
        return this.parent.senha === value;
    })

});

class DadosCadastrais extends Component {

    constructor (props) {
        super(props);
    };

    state = {

        nomeCompletoLogado: '',
        nomeUsuarioLogado: '',
        emailLogado: '',
        estadoLogado: '',
        cidadeLogado: '',
        senhaLogado: '',
        urlPatch: '',
        logStatus: false

    };

    static navigationOptions = {

        title: 'Dados Cadastrais',
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
     * Método para pegar o nome do usuário passado da tela anterior e chamar método para pega-lo
     * @author Pedro Biasutti
     */
    componentDidMount() {

        const { navigation } = this.props;
        const nomeUsuario = navigation.getParam('nomeUsuario', 'erro nomeUsuario');

        this.setState({nomeUsuarioLogado: nomeUsuario});

        this.getUserByNomeUsuario(nomeUsuario);
    };

    /**
     * Método que retorna o usuário sendo passado seu nome do usuário.
     * @author Pedro Biasutti
     * @param nomeUsuario - nome do usuário logado
     */
    getUserByNomeUsuario = async (nomeUsuario) => {

        let nomeUsuarioLogado = nomeUsuario;
        let urlPatch = '';
        let nomeCompleto = '';
        let email = '';
        let estado = '';
        let cidade = '';
        let senha = '';
        let validation = 0;

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

            console.log('NÃO DEU ERRO NO GET USUARIO REGISTRATION');

            urlPatch = response.data._links.self.href;
            nomeCompleto = response.data.nomeCompleto;
            email = response.data.email;
            estado = response.data.estado;
            cidade = response.data.cidade;
            senha = response.data.senha;
            validation = 7;

        })
        .catch (function(error){

            console.log('DEU ERRO NO GET USUARIO REGISTRATION');

        })

        if ( validation === 7 ) {

            this.setState({
                nomeCompletoLogado: nomeCompleto,
                senhaLogado: senha,
                emailLogado: email,
                estadoLogado: estado,
                cidadeLogado: cidade,
                urlPatch: urlPatch,
                logStatus: true
            });

        } else {

            this.geraAlerta('O nome do usuário não existe no banco de dados.\n\n Favor alterar este campo de dados !');

        }


    };

    /**
     * Método para checar se o'nomeUsuario'  e o 'email' já foram salvos anteriormente no banco, por outro usuário.
     * Além disso checa se a senha antiga bate com a do usuário antigo
     * @author Pedro Biasutti
     * @param values - Dados que foram digitados no form.
     */
    validaAlteraDados = async (values) => {

        let validation = 0;

        await axios({
            method: 'get',
            url: urlValidaAlteraDados,
            params: {
                dados: values,
                nomeUsuarioAntigo: this.state.nomeUsuarioLogado
            },
            headers: { 
                'Cache-Control': 'no-store',
            }
        })
        .then (function(response) {
            console.log('NÃO DEU ERRO VALIDA ALTERA DADOS');
            validation = response.data;
        })
        .catch (function(error){
            console.log('DEU ERRO VALIDA ALTERA DADOS');
        })

        return validation;
    };

    /**
     * Método para verificar se existe tem algum problema com os dados do form.
     * Caso positivo, gera um alerta com o(s) itens problematicos.
     * Caso negativo, salva as variaveis.
     * @author Pedro Biasutti
     * @param validation - serve como parâmetro para a realização ou não do patch.
     * @param values - Dados que foram digitados no form.
     */
    validaForm = ( validation, values) => {

        if ( validation === 3) {

            this.geraAlerta('O nome do usuário pentence a outro usuário.\n\n Favor alterar este campo de dados !');

        } else if (validation === 5) {

            this.geraAlerta('O email já existe no banco de dados.\n\n Favor alterar este campo de dados !');

        } else if (validation === 7) {

            this.geraAlerta('A senha antiga não está correta.\n\n Favor alterar este campo de dados !');

        } else if ( validation === 8) {

            this.geraAlerta('O nome do usuário e o email já existem no banco de dados.\n\n Favor alterar estes campos de dados !');

        } else if ( validation === 10) {

            this.geraAlerta('O nome do usuário já existe e a senha antiga não está correta.\n\n Favor alterar estes campos de dados !');

        } else if ( validation === 12) {

            this.geraAlerta('O email já existe e a senha antiga não está correta.\n\n Favor alterar estes campos de dados !');

        } else if ( validation === 15) {

            this.geraAlerta('O nome do usuário e o email já existem no banco de dados e a senha antiga não está correta.\n\n Favor alterar estes campos de dados !');

        } else {

            this.setState({
                nomeCompletoLogado: values.nomeCompleto,
                nomeUsuarioLogado: values.nomeUsuario,
                estadoLogado: values.estado,
                cidadeLogado: values.cidade,
                senhaLogado: values.senha,
                emailLogado: values.email
            });

            console.warn(JSON.stringify(values));
            
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
            },
            headers: { 
                'Cache-Control': 'no-store',
            }
        })
        .then (function(response) {
            console.log('DEU ERRO ATUALIZA USER');
            httpStatus = response.status;
        })
        .catch (function(error){
            console.log('NÃO DEU ERRO ATUALIZA USER');
        })

        return httpStatus;

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
        )
        
    };

    render () {

        return (
            <ScrollView>
                <SafeAreaView style = {{ marginTop: 20 }}>

                    <View style = {{alignItems: 'center', justifyContent: 'center', marginBottom:30}}>
                        <Text style = {styles.headers}>Alterar Dados Cadastrais</Text>
                    </View>

                    { this.state.logStatus &&
                        <Formik
                        initialValues={{
                            nomeCompleto: this.state.nomeCompletoLogado,
                            nomeUsuario: this.state.nomeUsuarioLogado,
                            email: this.state.emailLogado,
                            estado: this.state.estadoLogado,
                            cidade: this.state.cidadeLogado,
                            senhaAntiga: '',
                            senha: '',
                            confirmaSenha: '',
                        }}
                        // initialValues={{
                        //     nomeCompleto: this.state.nomeCompletoLogado,
                        //     nomeUsuario: this.state.nomeUsuarioLogado,
                        //     email: this.state.emailLogado,
                        //     senhaAntiga: '123456',
                        //     senha: 'asdasd',
                        //     confirmaSenha: 'asdasd',
                        // }}
                        onSubmit =  { async (values, actions) => {
                            
                            let validation = 0;
                            let httpStatus = 0;

                            // Verifica se os dados estão ok
                            await this.validaAlteraDados(values);
                            validation = await this.validaAlteraDados(values);

                            // Pela resposta do "validaAlteraDados" gera ou não um alerta
                            await this.validaForm(validation, values);

                            if ( validation == 0 ) {

                                // httpStatus = await this.atualizaUser();

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

                            actions.setSubmitting(false);



                        }}
                        // validateOnBlur= {false}
                        validateOnChange = {false}
                        validationSchema = {validationSchema}
                        >
                            {formikProps => (
                                <React.Fragment>

                                    <View>

                                        <View style = {styles.containerStyle}>
                                            <Text style = {styles.labelStyle}>Nome</Text>
                                            <TextInput
                                                placeholder = {this.state.nomeCompletoLogado}
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
                                                placeholder = {this.state.nomeUsuarioLogado}
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
                                                placeholder = {this.state.emailLogado}
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
                                                placeholder = {this.state.estadoLogado}
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
                                                placeholder = {this.state.cidadeLogado}
                                                style = {styles.inputStyle}
                                                onChangeText = {formikProps.handleChange('cidade')}
                                                onBlur = {formikProps.handleBlur('cidade')}
                                                onSubmitEditing = {() => { this.senhaAntiga.focus() }}
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
                                            <Text style = {styles.labelStyle}>Senha{'\n'}antiga</Text>
                                            <TextInput
                                                placeholder='senha123'
                                                style = {styles.inputStyle}
                                                onChangeText = {formikProps.handleChange('senhaAntiga')}
                                                onBlur = {formikProps.handleBlur('senhaAntiga')}
                                                secureTextEntry
                                                onSubmitEditing = {() => { this.senha.focus() }}
                                                ref = {(ref) => { this.senhaAntiga = ref; }}
                                                returnKeyType = { "next" }
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

                                    <View>

                                        <View style = {styles.containerStyle}>
                                            <Text style = {styles.labelStyle}>Senha</Text>
                                            <TextInput
                                                placeholder='senha456'
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
                                                placeholder='senha456'
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

                                    <View style = {{alignItems: 'center', marginTop: 30, marginBottom: 30}}>
                                    {formikProps.isSubmitting ? (
                                        <View style = {{scaleX: 1.5, scaleY: 1.5}}>
                                            <ActivityIndicator/>
                                        </View>
                                        ) : (
                                        <View style = {{flexDirection: 'column', flex: 1, width: '50%'}}>

                                            <StyledButton onPress={formikProps.handleSubmit}>
                                                Alterar
                                            </StyledButton>

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

export default DadosCadastrais;

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
    
});
