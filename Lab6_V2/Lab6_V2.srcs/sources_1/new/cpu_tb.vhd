library ieee;
use ieee.std_logic_1164.all;

entity cputb is
end entity;

architecture behav of cputb is
component cpu is 
    Port(
         reset : in std_logic ;
         cpuclk : in std_logic ;
         mdin : in std_logic_vector (15 downto 0);
         outputtt : out std_logic_vector(15 downto 0) 
            );
end component ;

signal clksignal,reset : std_logic ;
signal input : std_logic_vector (15 downto 0);
constant clk_period : time := 10 ns;

begin

Processor : cpu port map (reset => reset, cpuclk => clksignal, mdin => input);

clk_process :process
    begin
        clksignal  <= '0';
        wait for clk_period/2;
        clksignal  <= '1';
        wait for clk_period/2;
end process;

stim_proc: process
    begin
        reset <= '0';
        
        --Input mvi
        input <= "0000000001000000" ;
        wait for 35ns; 
        input <= "0101010101010101";
        wait for 30ns; 
        --Input mv
        input <= "0000000000001000";
        wait for 95ns;  
        --Input mvi
        input <= "0000000001000000";
        wait for 35ns;
        input <= "1010101010101010";
        wait for 30ns;
        --Change the code accordingly for the required operation.
        --performing "not" on R0 and storing it to R0
        input <= "0000000111000001";
        wait for 180ns;
        reset <= '1';
        input <= (others => 'U');
        wait ;
        

end process;

end architecture;