library IEEE;
use IEEE.STD_LOGIC_1164.ALL;

entity cu is
  Port (
        reset : in std_logic ;
        clk : in std_logic ;
        ir : in std_logic_vector (8 downto 0);
        muxval: out std_logic ;
        muxaddr : out std_logic_vector (3 downto 0);
        aluop : out std_logic_vector (2 downto 0);
        aluen : out std_logic
     );
end entity ;

architecture Behavioral of cu is
signal addrx,addry : std_logic_vector (3 downto 0);
signal instruction : std_logic_vector (2 downto 0);
signal tmuxaddr : std_logic_vector (3 downto 0);
signal tmuxval : std_logic ;
signal cc : std_logic := '0';
type statetype is (s00,s0,s1,s2,s3,s4,s5,s6,s7,s8,s9);
signal st,nst : statetype := s00;
begin
addry <= "0"&ir(2 downto 0);
addrx <= "0"&ir(5 downto 3);
instruction <= ir(8 downto 6);
process(instruction ,addrx , addry, st, tmuxaddr , tmuxval, nst )
begin
case st is
    when s00 =>
        nst <= s0;
    when s0 =>
        tmuxval <= '1';
        tmuxaddr  <= "1000";
        nst <= s1;
    when s1 =>
        tmuxval <= '0';
        tmuxaddr <= "1011";
        case instruction is 
            when "000" =>
                nst <= s6;
            when "001" =>
                nst <= s2;
            when "010" =>
                nst <= s4;
            when "011" =>
                nst <= s4;
            when "100" =>
                nst <= s4;
            when "101" =>
                nst <= s4;
            when "110" =>
                nst <= s4;
            when "111" =>
                nst <= s4;
            when others =>
                nst <= s0;
        end case;
    when s2 =>
        tmuxval <= '1';
        tmuxaddr <= "1000";
        nst <= s3;
    when s3 =>
        tmuxval <= '0';
        tmuxaddr <= addrx ;
        nst <= s0;
    when s4 =>
        tmuxval <= '1';
        tmuxaddr <= addrx;
        nst <= s5;
    when s5 =>
        tmuxval <= '0';
        tmuxaddr <= "1001";
        nst <= s6;
    when s6 =>
        tmuxval <= '1';
        tmuxaddr <= addry;
        case instruction is 
            when "000" =>
                nst <= s9;
            when others =>
                nst <= s7;
        end case;
    when s7 =>
        aluen <= '1';
        aluop <= instruction ;
        tmuxval <= '1';
        tmuxaddr <= "1010";
        tmuxval <= 'Z';
        nst <= s8;
    when s8 =>
        aluen <='0';
        tmuxval <= '1';
        tmuxaddr <= "1010";
        nst <= s9;
    when s9 =>
        tmuxval <= '0';
        tmuxaddr <= addrx;
        nst <= s0;
    when others =>
        nst <= s0;
end case;
end process;
muxaddr <= tmuxaddr;
muxval <= tmuxval ; 
clk_proc : process
begin 
wait until (clk'event and clk='1');
    if reset  = '1' then
        st <= s00;
    elsif cc = '0' then
        st <= nst;
    end if;
    cc <= not cc; 
end process ;
end Behavioral;
