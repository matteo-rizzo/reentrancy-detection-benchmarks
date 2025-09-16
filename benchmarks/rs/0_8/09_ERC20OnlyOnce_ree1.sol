// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

interface IERC20 {
    function totalSupply() external view returns (uint256);
    function balanceOf(address account) external view returns (uint256);
    function transfer(address to, uint256 amount) external returns (bool);
    function allowance(address owner, address spender) external view returns (uint256);
    function approve(address spender, uint256 amount) external returns (bool);
    function transferFrom(address from, address to, uint256 amount) external returns (bool);
    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);
}

contract C {

    mapping (address => bool) private donated;

    function donateTokens(address token, address to, uint256 amount) public {
        require(!donated[msg.sender]);
        require(IERC20(token).balanceOf(msg.sender) >= amount * 2, "Need at least double to donate");
        bool success = IERC20(token).transferFrom(msg.sender, to, amount);       
        donated[msg.sender] = success;
        require(success, "Donation failed");
    }
}
