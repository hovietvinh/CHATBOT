const initialState = {
    accountInfo:{},
    token:""
};

const AccountReducer = (state = initialState, action) => {
    switch (action.type) {
        case "":
            return {
                ...state,
                token:action.user_token,
                userInfo:action.data
            };
        
       
        
        
        
        default:
            return state;
    }
}

export default AccountReducer;
