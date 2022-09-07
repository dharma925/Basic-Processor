library ieee;
use ieee.std_logic_1164.ALL;
use ieee.std_logic_unsigned.ALL ;

entity  alu is
port(
        en: in std_logic ;
        a: in std_logic_vector (15 downto 0);
        b: in std_logic_vector (15 downto 0);
        op : in std_logic_vector (2 downto 0);
        output : out std_logic_vector (15 downto 0)
        );
end entity;

architecture behav of alu is

signal t16 : std_logic_vector (15 downto 0);
begin
process (a,b,op,en)
begin
if en = '1' then
case op is 
    when "010" =>
        t16 <= a+b;
    when "011" =>
        t16 <= a-b;
    when "100" =>
        t16 <= a and b;
    when "101" =>
        t16 <= a or b;
    when "110" =>
        t16 <= a xor b;
    when "111" =>
        t16 <= not a;
    when others=>
        t16 <= (others =>'Z');
end case;
end if;


end process ;

output <= t16;

end behav ;