const initialState = {
    userInfo:{},
    token:""
};

const UserReducer = (state = initialState, action) => {
    switch (action.type) {
        case "USER_LOGIN":
            return {
                ...state,
                token:action.user_token,
                userInfo:action.data
            };
        case "USER_SET":
            return {
                ...state,
                userInfo:action.data,
                token:action.user_token
            }

        case "USER_LOGOUT":
            return {
                ...state,
                userInfo:{},
                token:""
            }
       
        
        
        
        default:
            return state;
    }
}

export default UserReducer;
